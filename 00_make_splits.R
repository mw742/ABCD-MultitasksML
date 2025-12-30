#!/usr/bin/env Rscript

## 00_make_splits.R
## Goal: subject-level train/val/test split with ZERO IID overlap.
## Output:
##   splits/ids_train.tsv, splits/ids_val.tsv, splits/ids_test.tsv
##   splits/split_manifest.tsv

suppressPackageStartupMessages({
  library(data.table)
})

message_ts <- function(...) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste0(..., collapse=""))
}

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(key, default=NULL) {
  hit <- grep(paste0("^--", key, "="), args, value=TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", key, "="), "", hit[1])
}

SEED     <- as.integer(get_arg("seed", "12345"))
P_TRAIN  <- as.numeric(get_arg("p_train", "0.70"))
P_VAL    <- as.numeric(get_arg("p_val", "0.15"))
# Test is remainder: 1 - P_TRAIN - P_VAL
STOP_ON_DUP <- as.integer(get_arg("stop_on_duplicate_iid", "1"))  # 1=yes

if (!is.finite(P_TRAIN) || !is.finite(P_VAL) || P_TRAIN <= 0 || P_VAL <= 0 || (P_TRAIN + P_VAL) >= 1) {
  stop("Bad split proportions. Need p_train>0, p_val>0, p_train+p_val<1. Got: ", P_TRAIN, " / ", P_VAL)
}

set.seed(SEED)

WORKDIR <- getwd()
OUTDIR  <- file.path(WORKDIR, "splits")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

## ------------------------------------------------------------
## 1) Source of subject IDs
## ------------------------------------------------------------
id_source <- file.path(WORKDIR, "../Env_surv/cov_timevarying_filtered.tsv")
if (!file.exists(id_source)) {
  id_source2 <- file.path(WORKDIR, "cov_timevarying_filtered.tsv")
  if (!file.exists(id_source2)) stop("Cannot find ID source. Tried: ", id_source, " and ", id_source2)
  id_source <- id_source2
}

message_ts("Reading ID source: ", id_source)
dt <- fread(id_source)

need <- c("FID","IID")
if (!all(need %in% names(dt))) stop("ID source must contain FID and IID.")

ids <- unique(dt[, .(FID, IID)])
ids[, FID := as.character(FID)]
ids[, IID := as.character(IID)]
ids <- ids[!is.na(IID) & IID != ""]

message_ts("Total subjects: ", nrow(ids))

## check: IID should map to one FID (usually FID==IID anyway)
dup_iid <- ids[, .N, by=IID][N > 1]
if (nrow(dup_iid) > 0) {
  msg <- paste0("Found ", nrow(dup_iid), " IIDs with multiple rows in (FID,IID). Example IID=", dup_iid$IID[1])
  if (STOP_ON_DUP == 1L) stop(msg) else message_ts("[warn] ", msg)
  # Make it unique on IID if you want:
  ids <- unique(ids, by="IID")
  message_ts("After unique(by=IID): ", nrow(ids))
}

## ------------------------------------------------------------
## 2) Split (SEQUENTIAL sampling without replacement)
## ------------------------------------------------------------
N <- nrow(ids)
n_train <- floor(P_TRAIN * N)
n_val   <- floor(P_VAL * N)

# single permutation -> guaranteed disjoint sets
perm <- sample.int(N, size=N, replace=FALSE)

train_idx <- perm[1:n_train]
val_idx   <- perm[(n_train+1):(n_train+n_val)]
test_idx  <- perm[(n_train+n_val+1):N]

train <- ids[train_idx]
val   <- ids[val_idx]
test  <- ids[test_idx]

## ------------------------------------------------------------
## 3) Hard overlap QC
## ------------------------------------------------------------
inter_tv <- intersect(train$IID, val$IID)
inter_tt <- intersect(train$IID, test$IID)
inter_vt <- intersect(val$IID, test$IID)

if (length(inter_tv) + length(inter_tt) + length(inter_vt) > 0) {
  stop("IID overlap across splits detected (bug). ",
       "train∩val=", length(inter_tv),
       " train∩test=", length(inter_tt),
       " val∩test=", length(inter_vt))
}

message_ts("Split sizes: train=", nrow(train), " val=", nrow(val), " test=", nrow(test))

## ------------------------------------------------------------
## 4) Write outputs
## ------------------------------------------------------------
f_train <- file.path(OUTDIR, "ids_train.tsv")
f_val   <- file.path(OUTDIR, "ids_val.tsv")
f_test  <- file.path(OUTDIR, "ids_test.tsv")

fwrite(train, f_train, sep="\t")
fwrite(val,   f_val,   sep="\t")
fwrite(test,  f_test,  sep="\t")

manifest <- data.table(
  param = c("seed","p_train","p_val","p_test","n_total","n_train","n_val","n_test","id_source"),
  value = c(SEED, P_TRAIN, P_VAL, 1 - P_TRAIN - P_VAL, N, nrow(train), nrow(val), nrow(test), id_source)
)
fwrite(manifest, file.path(OUTDIR, "split_manifest.tsv"), sep="\t")

message_ts("Wrote: ", f_train)
message_ts("Wrote: ", f_val)
message_ts("Wrote: ", f_test)
message_ts("DONE.")

