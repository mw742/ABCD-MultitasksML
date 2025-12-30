#!/usr/bin/env Rscript
## 01_build_mtl_dataset_baseline_horizon.R 
##
## Build a "baseline + horizon" static dataset for multi-task learning.
## - Baseline features: first observed (min time_months) row per subject from cov_timevarying_filtered.tsv
## - Features: candidate exposures + core covars + PRS
## - Labels (per trait):
##     y_<trait> in {0,1,NA}
##     m_<trait> in {0,1} mask for whether label is observed for horizon classification
## - Splits: uses splits.tsv (FID IID split) OR splits/ids_train.tsv etc
##
## IMPORTANT: phenotype *_survival.tsv "time" may be AGE-in-months at event/censor (ABCD-like).
## This script automatically converts to FOLLOW-UP months since baseline:
##   t_followup = time_abs - baseline_age_months
##
suppressPackageStartupMessages({
  library(data.table)
})

ts_msg <- function(...) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste0(..., collapse=""))
}

## -----------------------------
## CLI args (no extra deps)
## -----------------------------
args <- commandArgs(trailingOnly=TRUE)
get_arg <- function(key, default=NULL) {
  hit <- grep(paste0("^--", key, "="), args, value=TRUE)
  if (length(hit)==0) return(default)
  sub(paste0("^--", key, "="), "", hit[1])
}

WORKDIR <- get_arg("workdir", getwd())
ENV_SURV_DIR <- get_arg("env_surv_dir",
  "/gpfs/group/ehlers/mwei/ABCD_2/ABCD/work_provided_assets/Env_surv"
)
COV_FILE <- get_arg("cov_file", file.path(ENV_SURV_DIR, "cov_timevarying_filtered.tsv"))
CAND_FILE <- get_arg("cand_file", file.path(ENV_SURV_DIR, "candidate_exposures_shared.tsv"))

SPLITS_TSV <- get_arg("splits", file.path(WORKDIR, "splits.tsv"))
SPLITS_DIR <- get_arg("splits_dir", file.path(WORKDIR, "splits"))

PHEN_DIR <- get_arg("phen_dir", "/gpfs/group/ehlers/mwei/ABCD/init_phenos")
OUT_DIR <- get_arg("outdir", file.path(WORKDIR, "results"))

HORIZON_MONTHS <- as.numeric(get_arg("horizon_months", "48"))
if (!is.finite(HORIZON_MONTHS) || HORIZON_MONTHS <= 0) stop("Bad --horizon_months: ", HORIZON_MONTHS)

dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

## -----------------------------
## Project-specific config
## -----------------------------
TRAITS <- c("alcohol","nicotine","cannabis","any")
PHEN_FILES <- setNames(file.path(PHEN_DIR, paste0(TRAITS, "_survival.tsv")), TRAITS)

CORE_COVARS <- c("sex","baseline_age","age","site", paste0("PC",1:20))

PRS_PATHS <- list(
  PRS_AUD      = "/gpfs/group/ehlers/mwei/ABCD_2/ABCD/work_provided_assets/GWAS_sud/PRS_scores_PRScsx/PAU_multi_PRS_two_col.tsv",
  PRS_CanUD    = "/gpfs/group/ehlers/mwei/ABCD_2/ABCD/work_provided_assets/GWAS_sud/Levey2023_CanUD/prs_scores_prscsx/CanUD_EURAFR_META_PRS_two_col.tsv",
  PRS_nicotine = "/gpfs/group/ehlers/mwei/ABCD_2/ABCD/work_provided_assets/GWAS_sud1/prs_scores/nicotine_PRS_two_col.tsv",
  PRS_anySUD   = "/gpfs/group/ehlers/mwei/ABCD_2/ABCD/work_provided_assets/GWAS_sud1/prs_scores/anySUD_PRS_two_col.tsv"
)

## -----------------------------
## Robust helpers
## -----------------------------
assert_no_dup_names <- function(dt, where="data.table") {
  dup <- names(dt)[duplicated(names(dt))]
  if (length(dup) > 0) stop(where, " has duplicated column names: ", paste(unique(dup), collapse=", "))
}

safe_merge_by_id <- function(x, y, all.x=TRUE, all.y=FALSE) {
  assert_no_dup_names(x, "Left table before merge")
  assert_no_dup_names(y, "Right table before merge")

  by <- c("FID","IID")
  if (!all(by %in% names(x))) stop("Left table missing keys: FID/IID")
  if (!all(by %in% names(y))) stop("Right table missing keys: FID/IID")

  dup <- intersect(setdiff(names(x), by), setdiff(names(y), by))
  if (length(dup) > 0) {
    stop("Refusing merge: duplicate non-key columns would be created: ", paste(dup, collapse=", "))
  }

  out <- merge(x, y, by=by, all.x=all.x, all.y=all.y)
  assert_no_dup_names(out, "Merged table")
  out
}

safe_read_score_table <- function(path, score_name) {
  if (!file.exists(path)) {
    ts_msg("[warn] PRS file missing, skipping: ", path)
    return(NULL)
  }

  dt <- tryCatch(fread(path, fill=TRUE), error=function(e) NULL)
  if (is.null(dt) || nrow(dt)==0) {
    dt <- fread(path, header=FALSE, fill=TRUE)
    if (ncol(dt) < 2) stop("PRS score file must have >=2 columns: ", path)
  }

  setnames(dt, names(dt), make.names(names(dt), unique=TRUE))

  if (all(c("FID","IID") %in% names(dt))) {
    score_cols <- setdiff(names(dt), c("FID","IID"))
    if (length(score_cols) != 1) {
      stop("PRS file must have exactly one score column besides FID/IID. Found: ",
           paste(score_cols, collapse=", "), " in ", path)
    }
    dt[, FID := as.character(FID)]
    dt[, IID := as.character(IID)]
    dt[, (score_cols) := as.numeric(dt[[score_cols]])]
    out <- dt[, .(FID, IID, score = dt[[score_cols]])]
    setnames(out, "score", score_name)
    return(out)
  }

  if ("IID" %in% names(dt) && !"FID" %in% names(dt)) {
    score_cols <- setdiff(names(dt), "IID")
    if (length(score_cols) != 1) {
      stop("PRS file with IID must have exactly one score column. Found: ",
           paste(score_cols, collapse=", "), " in ", path)
    }
    dt[, IID := as.character(IID)]
    dt[, FID := IID]
    dt[, (score_cols) := as.numeric(dt[[score_cols]])]
    out <- dt[, .(FID, IID, score = dt[[score_cols]])]
    setnames(out, "score", score_name)
    return(out)
  }

  if (ncol(dt) == 2) {
    id <- as.character(dt[[1]])
    sc <- suppressWarnings(as.numeric(dt[[2]]))
    out <- data.table(FID=id, IID=id)
    out[, (score_name) := sc]
    return(out)
  }

  if (ncol(dt) >= 3) {
    fid <- as.character(dt[[1]])
    iid <- as.character(dt[[2]])
    sc  <- suppressWarnings(as.numeric(dt[[3]]))
    out <- data.table(FID=fid, IID=iid)
    out[, (score_name) := sc]
    return(out)
  }

  stop("Unrecognized PRS format for file: ", path)
}

detect_col <- function(dt, candidates) {
  nms <- names(dt)
  low <- tolower(nms)
  cand_low <- tolower(candidates)
  idx <- match(cand_low, low)
  idx <- idx[!is.na(idx)]
  if (length(idx) == 0) return(NA_character_)
  nms[idx[1]]
}

parse_event_robust <- function(x) {
  # returns integer vector in {0,1,NA}
  s <- tolower(trimws(as.character(x)))
  s[s == ""] <- NA_character_
  e <- rep(NA_integer_, length(s))

  e[s %in% c("1","true","t","yes","y","event","case","positive","pos","fail","failure")] <- 1L
  e[s %in% c("0","false","f","no","n","censored","censor","control","negative","neg","ok")] <- 0L

  suppressWarnings({
    sn <- as.numeric(s)
    ok <- is.na(e) & !is.na(sn)
    e[ok] <- as.integer(round(sn[ok]))
  })

  # Keep only 0/1; others -> NA
  e[!(e %in% c(0L,1L))] <- NA_integer_
  e
}

get_baseline_age_months <- function(base_dt) {
  # Prefer interview_age if present (ABCD often months)
  if ("interview_age" %in% names(base_dt)) {
    x <- suppressWarnings(as.numeric(base_dt[["interview_age"]]))
    # heuristic: if median < 40, likely years -> convert; else keep
    md <- median(x, na.rm=TRUE)
    if (is.finite(md) && md < 40) x <- x * 12
    return(x)
  }

  # baseline_age may be years or months; convert if needed
  if ("baseline_age" %in% names(base_dt)) {
    x <- suppressWarnings(as.numeric(base_dt[["baseline_age"]]))
    md <- median(x, na.rm=TRUE)
    if (is.finite(md) && md < 40) x <- x * 12
    return(x)
  }

  # age fallback
  if ("age" %in% names(base_dt)) {
    x <- suppressWarnings(as.numeric(base_dt[["age"]]))
    md <- median(x, na.rm=TRUE)
    if (is.finite(md) && md < 40) x <- x * 12
    return(x)
  }

  rep(NA_real_, nrow(base_dt))
}

convert_time_to_followup_months <- function(t_abs, baseline_age_months) {
  # Return list(t = chosen_time, used_rel = TRUE/FALSE)
  t_abs <- suppressWarnings(as.numeric(t_abs))
  bam   <- suppressWarnings(as.numeric(baseline_age_months))

  used_rel <- FALSE
  t <- t_abs

  if (any(is.finite(bam))) {
    delta <- t_abs - bam
    # Choose delta if it looks like a follow-up time: mostly >=0 and not absurd.
    # This works for ABCD where t_abs is age-months ~100-200 and baseline age-months ~110-140.
    frac_ok <- mean(is.finite(delta) & delta >= 0 & delta <= 300, na.rm=TRUE)
    md_ok   <- median(delta, na.rm=TRUE)

    if (is.finite(frac_ok) && frac_ok > 0.5 && is.finite(md_ok) && md_ok >= 0 && md_ok <= 300) {
      t <- delta
      used_rel <- TRUE
    }
  }

  list(t=t, used_rel=used_rel)
}

make_labels_by_horizon <- function(ph, horizon_months) {
  ph <- as.data.table(ph)

  if (!all(c("FID","IID") %in% names(ph))) stop("Phenotype file missing FID/IID")

  event_col <- if ("event" %in% names(ph)) "event" else detect_col(ph, c("event","status","case","y","outcome","fail","endpoint"))
  time_col  <- if ("time"  %in% names(ph)) "time"  else detect_col(ph, c("time","time_months","t","t_months","event_time","time_to_event","months_to_event","stop","duration","followup_months"))

  if (is.na(event_col)) stop("Could not find event/status column in phenotype file. Columns: ", paste(names(ph), collapse=", "))
  if (is.na(time_col))  stop("Could not find time column in phenotype file. Columns: ", paste(names(ph), collapse=", "))

  e <- parse_event_robust(ph[[event_col]])

  # Convert time:
  # If baseline_age_months exists, and time looks like age-months, use (time - baseline_age_months).
  if (!("baseline_age_months" %in% names(ph))) {
    # keep raw
    t_use <- suppressWarnings(as.numeric(ph[[time_col]]))
    used_rel <- FALSE
  } else {
    conv <- convert_time_to_followup_months(ph[[time_col]], ph[["baseline_age_months"]])
    t_use <- conv$t
    used_rel <- isTRUE(conv$used_rel)
  }

  H <- as.numeric(horizon_months)
  if (!is.finite(H) || H <= 0) stop("Bad horizon_months: ", horizon_months)

  # Correct horizon classification for survival data:
  # - If event happens by H: y=1, m=1
  # - If subject is event-free through H (either censored at t>=H OR event after H): y=0, m=1
  # - If censored before H (event=0 & t<H) or missing: y=NA, m=0
  tfin <- is.finite(t_use)
  m <- rep(0L, length(e))
  y <- rep(NA_integer_, length(e))

  # Observed through horizon
  obs_through_H <- tfin & !is.na(e) & (t_use >= H)
  m[obs_through_H] <- 1L
  y[obs_through_H] <- 0L

  # Event by horizon
  ev_by_H <- tfin & (e == 1L) & (t_use < H)
  m[ev_by_H] <- 1L
  y[ev_by_H] <- 1L

  censored_before <- sum((e == 0L) & tfin & (t_use < H), na.rm=TRUE)

  dt <- data.table(
    FID = as.character(ph[["FID"]]),
    IID = as.character(ph[["IID"]]),
    y   = y,
    m   = as.integer(m)
  )

  n_obs <- sum(dt$m==1L, na.rm=TRUE)
  n_pos <- sum(dt$y==1L & dt$m==1L, na.rm=TRUE)
  ts_msg("[label-QA] event_col=", event_col, " time_col=", time_col,
         ifelse(used_rel, " (using time-baseline_age_months)", " (using raw time)"),
         " | obs=", n_obs, " pos=", n_pos,
         " prev=", ifelse(n_obs>0, round(n_pos/n_obs, 4), NA))

  list(
    dt = dt,
    censored_before_horizon = as.integer(censored_before),
    used_time = TRUE,
    time_col = time_col
  )
}

load_splits <- function(splits_tsv, splits_dir) {
  if (file.exists(splits_tsv)) {
    sp <- fread(splits_tsv)
    if (!all(c("FID","IID","split") %in% names(sp))) stop("splits.tsv must have FID IID split")
    sp[, FID := as.character(FID)]
    sp[, IID := as.character(IID)]
    sp[, split := as.character(split)]
    return(sp[, .(FID, IID, split)])
  }

  f_train <- file.path(splits_dir, "ids_train.tsv")
  f_val   <- file.path(splits_dir, "ids_val.tsv")
  f_test  <- file.path(splits_dir, "ids_test.tsv")
  if (!all(file.exists(c(f_train, f_val, f_test)))) {
    stop("Missing splits.tsv and also missing one of splits/ids_{train,val,test}.tsv")
  }

  read_ids <- function(f, label) {
    x <- fread(f)
    if (!all(c("FID","IID") %in% names(x))) stop("Split file must have FID IID: ", f)
    x[, FID := as.character(FID)]
    x[, IID := as.character(IID)]
    x[, split := label]
    x[, .(FID, IID, split)]
  }

  rbindlist(list(
    read_ids(f_train, "train"),
    read_ids(f_val, "val"),
    read_ids(f_test, "test")
  ), use.names=TRUE)
}

## -----------------------------
## Load splits
## -----------------------------
ts_msg("Loading splits...")
splits <- load_splits(SPLITS_TSV, SPLITS_DIR)
ts_msg("Splits rows: ", nrow(splits), " | unique IID: ", uniqueN(splits$IID))
if (anyDuplicated(paste(splits$FID, splits$IID)) > 0) stop("Duplicate FID/IID rows in splits.")

## -----------------------------
## Load covariates and build baseline
## -----------------------------
if (!file.exists(COV_FILE)) stop("Missing covariate file: ", COV_FILE)
ts_msg("Reading covariates: ", COV_FILE)
cov <- fread(COV_FILE)
need_keys <- c("FID","IID","time_months")
if (!all(need_keys %in% names(cov))) stop("cov_timevarying_filtered.tsv must include FID IID time_months")

cov[, FID := as.character(FID)]
cov[, IID := as.character(IID)]
cov[, time_months := suppressWarnings(as.numeric(time_months))]

cov <- cov[paste(FID,IID) %in% paste(splits$FID, splits$IID)]
ts_msg("Cov rows after restricting to split IDs: ", nrow(cov))

setorder(cov, FID, IID, time_months)
ts_msg("Selecting baseline row per subject (min time_months)...")
base <- cov[, .SD[1], by=.(FID, IID)]
ts_msg("Baseline subjects: ", nrow(base))

# Baseline age months (for phenotype time conversion)
base_age <- base[, .(FID, IID)]
base_age[, baseline_age_months := get_baseline_age_months(base)]
ts_msg("Baseline age months: median=", round(median(base_age$baseline_age_months, na.rm=TRUE), 2),
       " | missing=", sum(!is.finite(base_age$baseline_age_months)))

## -----------------------------
## Candidate exposures
## -----------------------------
if (!file.exists(CAND_FILE)) stop("Missing candidate file: ", CAND_FILE)
cand <- fread(CAND_FILE)
if (!"term" %in% names(cand)) stop("candidate_exposures_shared.tsv must have column 'term'")
cand_terms <- unique(as.character(cand$term))

exposures <- intersect(cand_terms, names(base))
drop_like <- c("src_subject_id","eventname","interview_age","time_months")
exposures <- setdiff(exposures, intersect(exposures, drop_like))

is_num <- vapply(exposures, function(v) is.numeric(base[[v]]) || is.integer(base[[v]]), logical(1))
exposures <- exposures[is_num]
ts_msg("Candidate exposures found in baseline table: ", length(exposures))

core_present <- intersect(CORE_COVARS, names(base))
ts_msg("Core covars present: ", paste(core_present, collapse=", "))

feat_cols <- unique(c(core_present, exposures))
feat_cols <- feat_cols[feat_cols %in% names(base)]
dt_feat <- base[, c("FID","IID", feat_cols), with=FALSE]
assert_no_dup_names(dt_feat, "Features table")

## -----------------------------
## Merge PRS
## -----------------------------
ts_msg("Merging PRS tables ..")
prs_list <- list()
for (nm in names(PRS_PATHS)) {
  prs_dt <- safe_read_score_table(PRS_PATHS[[nm]], nm)
  if (!is.null(prs_dt)) prs_list[[nm]] <- prs_dt
}
if (length(prs_list) > 0) {
  prs_all <- Reduce(function(a,b) safe_merge_by_id(a,b, all.x=TRUE, all.y=TRUE), prs_list)
  dt_feat <- safe_merge_by_id(dt_feat, prs_all, all.x=TRUE, all.y=FALSE)
  ts_msg("After PRS merge: n=", nrow(dt_feat), " p=", ncol(dt_feat))
} else {
  ts_msg("No PRS loaded (all missing) — continuing.")
}

## -----------------------------
## Labels (multi-task)
## -----------------------------
ts_msg("Building labels by horizon_months=", HORIZON_MONTHS, " ...")
lab_all <- dt_feat[, .(FID, IID)]
censored_total <- 0L
used_time_any <- FALSE
time_cols_used <- c()

for (tr in names(PHEN_FILES)) {
  f <- PHEN_FILES[[tr]]
  if (!file.exists(f)) {
    ts_msg("[warn] Missing phenotype file for ", tr, ": ", f, " (skipping)")
    next
  }
  ts_msg("Reading phenotype: ", f)
  ph <- fread(f)
  ph[, FID := as.character(FID)]
  ph[, IID := as.character(IID)]

  # merge baseline age months into phenotype for time conversion
  ph <- merge(ph, base_age, by=c("FID","IID"), all.x=TRUE)

  res <- make_labels_by_horizon(ph, HORIZON_MONTHS)
  censored_total <- censored_total + res$censored_before_horizon
  used_time_any <- used_time_any || isTRUE(res$used_time)
  time_cols_used <- unique(c(time_cols_used, res$time_col))

  yname <- paste0("y_", tr)
  mname <- paste0("m_", tr)

  tmp <- res$dt[, .(FID, IID, y, m)]
  setnames(tmp, c("y","m"), c(yname, mname))

  lab_all <- safe_merge_by_id(lab_all, tmp, all.x=TRUE, all.y=FALSE)
}

label_cols <- grep("^y_", names(lab_all), value=TRUE)
mask_cols  <- grep("^m_", names(lab_all), value=TRUE)
if (length(label_cols) == 0) stop("No y_* labels were created. Check phenotype files.")
if (length(mask_cols)  == 0) stop("No m_* masks were created (unexpected).")

dt <- safe_merge_by_id(dt_feat, lab_all, all.x=TRUE, all.y=FALSE)

# Keep only subjects with at least 1 observed label (mask==1)
dt[, label_obs := rowSums(.SD == 1L, na.rm=TRUE), .SDcols=mask_cols]
dt <- dt[label_obs > 0]
dt[, label_obs := NULL]

# QA: stop if any trait has zero positives among observed labels
for (tr in TRAITS) {
  yc <- paste0("y_", tr)
  mc <- paste0("m_", tr)
  if (!(yc %in% names(dt)) || !(mc %in% names(dt))) next
  yobs <- dt[get(mc) == 1L, get(yc)]
  nobs <- length(yobs)
  npos <- sum(yobs == 1L, na.rm=TRUE)
  prev <- if (nobs > 0) npos / nobs else NA_real_
  ts_msg("[QA-final] ", tr, " obs=", nobs, " pos=", npos, " prev=", round(prev,4))
  if (npos == 0) stop("QA failed: zero positive labels for trait ", tr, " (check time conversion / phenotype columns)")
}

ts_msg("Total subjects after label mask filter: ", nrow(dt),
       " | Censored-before-horizon (raw count): ", censored_total)

## -----------------------------
## Attach split + write datasets
## -----------------------------
dt <- safe_merge_by_id(dt, splits, all.x=TRUE, all.y=FALSE)
if (any(is.na(dt$split))) stop("Some rows have missing split after merge. Splits file inconsistent with IDs.")

assert_no_dup_names(dt, "Final dataset")
if (anyDuplicated(paste(dt$FID, dt$IID)) > 0) stop("Duplicate subjects in final dataset.")

out_full  <- file.path(OUT_DIR, "mtl_dataset_full.tsv")
out_train <- file.path(OUT_DIR, "mtl_dataset_train.tsv")
out_val   <- file.path(OUT_DIR, "mtl_dataset_val.tsv")
out_test  <- file.path(OUT_DIR, "mtl_dataset_test.tsv")

ts_msg("Writing: ", out_full)
fwrite(dt, out_full, sep="\t", quote=FALSE, na="NA")
fwrite(dt[split=="train"], out_train, sep="\t", quote=FALSE, na="NA")
fwrite(dt[split=="val"],   out_val,   sep="\t", quote=FALSE, na="NA")
fwrite(dt[split=="test"],  out_test,  sep="\t", quote=FALSE, na="NA")

## -----------------------------
## Manifest
## -----------------------------
manifest <- data.table(
  param = c("script","timestamp","workdir","env_surv_dir","cov_file","cand_file","splits",
            "phen_dir","traits","horizon_months",
            "n_subjects_final","n_features",
            "n_label_cols","n_mask_cols",
            "used_time_labels","time_cols_used"),
  value = c("01_build_mtl_dataset_baseline_horizon.R",
            format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
            WORKDIR, ENV_SURV_DIR, COV_FILE, CAND_FILE,
            if (file.exists(SPLITS_TSV)) SPLITS_TSV else SPLITS_DIR,
            PHEN_DIR, paste(TRAITS, collapse=","),
            as.character(HORIZON_MONTHS),
            as.character(nrow(dt)),
            as.character(length(setdiff(names(dt), c("FID","IID","split", label_cols, mask_cols)))),
            as.character(length(label_cols)),
            as.character(length(mask_cols)),
            as.character(used_time_any),
            paste(unique(na.omit(time_cols_used)), collapse=","))
)

out_manifest <- file.path(OUT_DIR, "mtl_build_manifest.tsv")
fwrite(manifest, out_manifest, sep="\t", quote=FALSE, na="NA")

ts_msg("DONE. Wrote datasets to: ", OUT_DIR)

