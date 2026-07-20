import os
import numpy as np
import json
import random
import shutil
from pathlib import Path
import re
import pandas as pd

RANDOM_SEED = 20251226

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SOURCE_PATIENTS_DIR = "data_results/patients"
OUTPUT_DIR = "split_data"
HEALTH_STATS_FILE = "./health_statistics.txt"
###health_statistics.txt is the ECG records and the patient's affiliation relationship
###See:https://physionet.org/content/ptbdb/1.0.0/


TARGET_TRAIN = 55
TARGET_VAL = 11
TARGET_TEST = 13
Split_type = "patient"


def parse_health_stats(file_path):
    """
    Parse health_statistics.txt and return:
        patient_records: dict {patient_id: [record1, record2, ...]}
        total_cases: total number of cases
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Statistics file not found: {file_path}")

    patient_records = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_started = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-'):
            continue
        if 'Patient ID' in line or 'Health Patient Statistics' in line or 'Statistics' in line:
            data_started = True
            continue
        if not data_started:
            continue

        parts = re.split(r'\s{2,}', line)
        if len(parts) < 3:
            parts = line.split('\t')
        if len(parts) >= 3:
            patient = parts[0].strip()
            rec_str = parts[2].strip().strip('[]()')
            records = [r.strip() for r in rec_str.split(',') if r.strip()]
            if patient and records:
                patient_records[patient] = records
        else:
            continue

    total_cases = sum(len(recs) for recs in patient_records.values())
    print(f"[Parse] Read {len(patient_records)} patients, total cases {total_cases}")
    return patient_records, total_cases


def get_existing_records(source_dir):
    """Return set of existing record folder names in source directory"""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    return {d.name for d in source_path.iterdir() if d.is_dir()}


def split_patients_by_patient(patient_records, target_train, target_val, target_test):
    """
    Split patients by case count using greedy strategy:
    1. Shuffle all patients together randomly
    2. Try to add each patient to train set sequentially until target reached
    3. Continue adding remaining patients to validation set
    4. Remaining patients form test set
    Returns three lists, each element is (patient_id, [record1, ...]).
    """
    all_patients = list(patient_records.items())
    random.shuffle(all_patients)

    multi_count = sum(1 for _, recs in all_patients if len(recs) > 1)
    single_count = len(all_patients) - multi_count
    total_cases = sum(len(recs) for _, recs in all_patients)
    
    print(f"\n[Patient Distribution] Multi-case patients: {multi_count}, Single-case patients: {single_count}, Total cases: {total_cases}")

    train_patients = []
    train_cases = 0
    remaining_patients = []
    
    for pid, recs in all_patients:
        if train_cases + len(recs) <= target_train:
            train_patients.append((pid, recs))
            train_cases += len(recs)
        else:
            remaining_patients.append((pid, recs))

    actual_train = train_cases
    train_multi = sum(1 for _, recs in train_patients if len(recs) > 1)
    train_single = len(train_patients) - train_multi
    print(f"[Train Set] Cases: {actual_train}, Patients: {len(train_patients)} (multi-case {train_multi} + single-case {train_single})")

    val_patients = []
    val_cases = 0
    test_patients = []
    
    for pid, recs in remaining_patients:
        if val_cases + len(recs) <= target_val:
            val_patients.append((pid, recs))
            val_cases += len(recs)
        else:
            test_patients.append((pid, recs))

    actual_val = val_cases
    val_multi = sum(1 for _, recs in val_patients if len(recs) > 1)
    val_single = len(val_patients) - val_multi
    print(f"[Validation Set] Cases: {actual_val}, Patients: {len(val_patients)} (multi-case {val_multi} + single-case {val_single})")

    actual_test = sum(len(recs) for _, recs in test_patients)
    test_multi = sum(1 for _, recs in test_patients if len(recs) > 1)
    test_single = len(test_patients) - test_multi
    print(f"[Test Set] Cases: {actual_test}, Patients: {len(test_patients)} (multi-case {test_multi} + single-case {test_single})")

    train_patients = sorted(train_patients, key=lambda x: x[0])
    val_patients = sorted(val_patients, key=lambda x: x[0])
    test_patients = sorted(test_patients, key=lambda x: x[0])

    total_actual = actual_train + actual_val + actual_test
    expected_total = target_train + target_val + target_test
    if total_actual != expected_total:
        print(f"[Warning] Total cases {total_actual} does not match expected {expected_total}, please check.")

    train_pids = set(pid for pid, _ in train_patients)
    val_pids = set(pid for pid, _ in val_patients)
    test_pids = set(pid for pid, _ in test_patients)
    
    overlap_train_val = train_pids & val_pids
    overlap_train_test = train_pids & test_pids
    overlap_val_test = val_pids & test_pids
    
    if overlap_train_val:
        print(f"[Error] Following patients appear in both train and validation sets: {overlap_train_val}")
    if overlap_train_test:
        print(f"[Error] Following patients appear in both train and test sets: {overlap_train_test}")
    if overlap_val_test:
        print(f"[Error] Following patients appear in both validation and test sets: {overlap_val_test}")
    
    if not overlap_train_val and not overlap_train_test and not overlap_val_test:
        print("[Validation] All patients appear in only one set, split strategy is correct")

    print(f"\n[Split Result] Train: {len(train_patients)} patients, {actual_train} cases")
    print(f"              Validation: {len(val_patients)} patients, {actual_val} cases")
    print(f"              Test: {len(test_patients)} patients, {actual_test} cases")

    return train_patients, val_patients, test_patients


def split_patients_by_cases(patient_records, target_train, target_val, target_test):
    """
    Split patients by case count using greedy strategy:
    1. Shuffle multi-case patients, add to train sequentially until adding one more would exceed target,
       then use single-case patients to fill remaining slots.
    2. From remaining patients (unselected multi-case + all single-case), prioritize multi-case for validation,
       fill remainder with single-case.
    3. Remaining patients naturally form test set.
    Returns three lists, each element is (patient_id, [record1, ...]).
    """
    multi = {p: recs for p, recs in patient_records.items() if len(recs) > 1}
    single = {p: recs for p, recs in patient_records.items() if len(recs) == 1}

    print("\n[Multi-case Patients] The following patients have multiple records and will be prioritized for training:")
    multi_items = list(multi.items())
    for p, recs in multi_items[:10]:
        print(f"  {p} (total {len(recs)} cases): {', '.join(recs)}")
    if len(multi_items) > 10:
        print(f"  ... total {len(multi_items)} multi-case patients")
    multi_cases = sum(len(recs) for _, recs in multi_items)
    print(f"  Total multi-case patients: {len(multi_items)}, total cases: {multi_cases}\n")

    random.shuffle(multi_items)
    train_patients = []
    train_cases = 0
    for pid, recs in multi_items:
        if train_cases + len(recs) <= target_train:
            train_patients.append((pid, recs))
            train_cases += len(recs)
        else:
            break
    remaining_multi = multi_items[len(train_patients):]

    need_single = target_train - train_cases
    if need_single < 0:
        raise RuntimeError("Train set cases already exceed target, logical error")
    single_items = list(single.items())
    random.shuffle(single_items)
    if len(single_items) < need_single:
        raise ValueError(f"Insufficient single-case patients: need {need_single} to fill train set, only {len(single_items)} available.")
    train_single = single_items[:need_single]
    train_patients.extend(train_single)
    remaining_single = single_items[need_single:]

    actual_train = train_cases + need_single
    print(f"[Train Set] Cases: {actual_train} (multi-case {train_cases} + single-case {need_single})")

    remaining_all = remaining_multi + remaining_single
    random.shuffle(remaining_all)

    remaining_multi_only = [item for item in remaining_all if item[0] in dict(multi).keys()]
    remaining_single_only = [item for item in remaining_all if item[0] in dict(single).keys()]

    val_patients = []
    val_cases = 0
    for pid, recs in remaining_multi_only:
        if val_cases + len(recs) <= target_val:
            val_patients.append((pid, recs))
            val_cases += len(recs)
    need_val_single = target_val - val_cases
    if need_val_single < 0:
        raise RuntimeError("Validation set cases already exceed target, logical error")
    if len(remaining_single_only) < need_val_single:
        raise ValueError(f"Insufficient remaining single-case patients: need {need_val_single}, actual {len(remaining_single_only)}")
    temp_single = remaining_single_only[:]
    random.shuffle(temp_single)
    val_single_picked = temp_single[:need_val_single]
    val_patients.extend(val_single_picked)
    val_cases += need_val_single

    actual_val = val_cases
    print(f"[Validation Set] Cases: {actual_val}")

    selected_val_set = set(pid for pid, _ in val_patients)
    test_patients = [item for item in remaining_all if item[0] not in selected_val_set]
    actual_test = sum(len(recs) for _, recs in test_patients)

    print(f"[Test Set] Cases: {actual_test}")

    train_patients = sorted(train_patients, key=lambda x: x[0])
    val_patients = sorted(val_patients, key=lambda x: x[0])
    test_patients = sorted(test_patients, key=lambda x: x[0])

    total_actual = actual_train + actual_val + actual_test
    expected_total = target_train + target_val + target_test
    if total_actual != expected_total:
        print(f"[Warning] Total cases {total_actual} does not match expected {expected_total}, please check.")

    print(f"\n[Split Result] Train: {len(train_patients)} patients, {actual_train} cases")
    print(f"              Validation: {len(val_patients)} patients, {actual_val} cases")
    print(f"              Test: {len(test_patients)} patients, {actual_test} cases")

    return train_patients, val_patients, test_patients


def create_symlink_records(patient_list, split_name, source_dir, output_dir):
    """
    Create symbolic links for record folders to target split directory (no file copying, saves space).
    Returns list of patient info including window count for each record.
    """
    split_path = Path(output_dir) / split_name
    split_path.mkdir(parents=True, exist_ok=True)

    patient_info = []
    source_path = Path(source_dir)

    for pid, records in patient_list:
        records_info = []
        for rec in records:
            src_rec_dir = source_path / rec
            dst_rec_dir = split_path / rec
            if not src_rec_dir.exists():
                print(f"  Warning: Record {rec} for patient {pid} folder does not exist, skipping")
                continue
            src_ecg = src_rec_dir / "ecg_12lead.npy"
            src_vcg = src_rec_dir / "vcg.npy"
            if not src_ecg.exists() or not src_vcg.exists():
                print(f"  Warning: Record {rec} for patient {pid} missing .npy files, skipping")
                continue
            
            if dst_rec_dir.exists():
                if os.path.islink(dst_rec_dir):
                    os.remove(dst_rec_dir)
                else:
                    shutil.rmtree(dst_rec_dir)
            
            try:
                os.symlink(src_rec_dir, dst_rec_dir, target_is_directory=True)
            except OSError as e:
                print(f"  Warning: Failed to create symbolic link ({e}), falling back to file copy")
                dst_rec_dir.mkdir(exist_ok=True)
                shutil.copy2(src_ecg, dst_rec_dir / "ecg_12lead.npy")
                shutil.copy2(src_vcg, dst_rec_dir / "vcg.npy")
            
            ecg_data = np.load(src_ecg)
            vcg_data = np.load(src_vcg)
            records_info.append({
                "record": rec,
                "n_windows": ecg_data.shape[0],
                "ecg_shape": list(ecg_data.shape),
                "vcg_shape": list(vcg_data.shape)
            })
        if records_info:
            patient_info.append({
                "patient_id": pid,
                "records": records_info,
                "total_windows": sum(r["n_windows"] for r in records_info)
            })
        else:
            print(f"  Note: Patient {pid} has no valid records processed")
    return patient_info


def save_split_info(output_dir, train_info, val_info, test_info, target_cases, total_cases):
    info = {
        "split_target_cases": {
            "train": target_cases[0],
            "val": target_cases[1],
            "test": target_cases[2]
        },
        "random_seed": RANDOM_SEED,
        "total_patients": len(train_info)+len(val_info)+len(test_info),
        "total_cases": total_cases,
        "train": {
            "patients": train_info,
            "count": len(train_info),
            "total_windows": sum(p["total_windows"] for p in train_info),
            "total_cases": sum(len(p["records"]) for p in train_info)
        },
        "val": {
            "patients": val_info,
            "count": len(val_info),
            "total_windows": sum(p["total_windows"] for p in val_info),
            "total_cases": sum(len(p["records"]) for p in val_info)
        },
        "test": {
            "patients": test_info,
            "count": len(test_info),
            "total_windows": sum(p["total_windows"] for p in test_info),
            "total_cases": sum(len(p["records"]) for p in test_info)
        }
    }
    info_path = Path(output_dir) / "split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"[Save] Split info written to: {info_path}")


def main():
    print("=" * 70)
    print("MI Patient Dataset Split (by case count, greedy prioritize multi-case for training)")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Target: Train {TARGET_TRAIN} cases, Validation {TARGET_VAL} cases, Test {TARGET_TEST} cases")
    print("=" * 70)

    print("\n[1] Reading statistics file...")
    try:
        patient_records, total_cases = parse_health_stats(HEALTH_STATS_FILE)
    except Exception as e:
        print(f"Error: {e}")
        return

    if not patient_records:
        print("Error: No patient records after parsing statistics file, please check file format.")
        return

    print("\n[2] Checking original data directory...")
    try:
        existing_records = get_existing_records(SOURCE_PATIENTS_DIR)
    except Exception as e:
        print(f"Error: {e}")
        return

    filtered_records = {}
    for pid, recs in patient_records.items():
        valid_recs = [r for r in recs if r in existing_records]
        if valid_recs:
            filtered_records[pid] = valid_recs
        else:
            print(f"[Note] All records for patient {pid} are not in source directory, will be ignored")

    if not filtered_records:
        print("Error: No patients to process after filtering, please check paths.")
        return

    print(f"Actual processable patients: {len(filtered_records)}, total cases: {sum(len(r) for r in filtered_records.values())}")

    print("\n[3] Performing split...")
    if Split_type == "case":
        try:
            train_list, val_list, test_list = split_patients_by_cases(
                filtered_records, TARGET_TRAIN, TARGET_VAL, TARGET_TEST
            )
        except Exception as e:
            print(f"Split failed: {e}")
            return
    elif Split_type == "patient":
        try:
            train_list, val_list, test_list = split_patients_by_patient(
                filtered_records, TARGET_TRAIN, TARGET_VAL, TARGET_TEST
            )
        except Exception as e:
            print(f"Split failed: {e}")
            return
    else:
        print("Wrong split type")

    print("\n[4] Creating symbolic links to output directory...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        train_info = create_symlink_records(train_list, "train", SOURCE_PATIENTS_DIR, OUTPUT_DIR)
        val_info = create_symlink_records(val_list, "val", SOURCE_PATIENTS_DIR, OUTPUT_DIR)
        test_info = create_symlink_records(test_list, "test", SOURCE_PATIENTS_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Failed to create symbolic links: {e}")
        return

    print("\n[5] Saving split info...")
    save_split_info(OUTPUT_DIR, train_info, val_info, test_info,
                    (TARGET_TRAIN, TARGET_VAL, TARGET_TEST), total_cases)

    print("\n[6] Exporting patient IDs and record names to Excel...")
    
    train_pids = []
    train_records = []
    for p in train_info:
        for record in p['records']:
            train_pids.append(p['patient_id'])
            train_records.append(record)
    
    val_pids = []
    val_records = []
    for p in val_info:
        for record in p['records']:
            val_pids.append(p['patient_id'])
            val_records.append(record)
    
    test_pids = []
    test_records = []
    for p in test_info:
        for record in p['records']:
            test_pids.append(p['patient_id'])
            test_records.append(record)
    
    max_len = max(len(train_records), len(val_records), len(test_records))
    df = pd.DataFrame({
        'Train Patient ID': train_pids + [''] * (max_len - len(train_pids)),
        'Train Record': train_records + [''] * (max_len - len(train_records)),
        'Validation Patient ID': val_pids + [''] * (max_len - len(val_pids)),
        'Validation Record': val_records + [''] * (max_len - len(val_records)),
        'Test Patient ID': test_pids + [''] * (max_len - len(test_pids)),
        'Test Record': test_records + [''] * (max_len - len(test_records))
    })
    
    excel_path = Path(OUTPUT_DIR) / "patient_split.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"    Saved: {excel_path}")

    print("\n" + "=" * 70)
    print("Split completed!")
    print(f"Train: {len(train_info)} patients, {sum(len(p['records']) for p in train_info)} cases, {sum(p['total_windows'] for p in train_info)} windows")
    print(f"Validation: {len(val_info)} patients, {sum(len(p['records']) for p in val_info)} cases, {sum(p['total_windows'] for p in val_info)} windows")
    print(f"Test: {len(test_info)} patients, {sum(len(p['records']) for p in test_info)} cases, {sum(p['total_windows'] for p in test_info)} windows")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()