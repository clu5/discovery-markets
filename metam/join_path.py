import pandas as pd
import random
import logging
from join_column import JoinColumn

# Configure logging
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

def load_dataframe(tbl_name, path, data_dic):
    """
    Loads a DataFrame from a CSV file if not already loaded.
    Updates the data_dic cache with the loaded DataFrame.
    """
    if tbl_name not in data_dic:
        try:
            df = pd.read_csv(f"{path}/{tbl_name}", low_memory=False)
            data_dic[tbl_name] = df
            logging.info(f"Loaded dataset {tbl_name} with shape {df.shape}")
        except Exception as e:
            logging.warning(f"Failed to load {tbl_name}: {e}")
            return None
    return data_dic.get(tbl_name)

def should_skip_join(jp, ignore_lst, size_dic, excluded_tables):
    """
    Checks whether a join path should be skipped based on table exclusion lists and table sizes.
    """
    left_tbl = jp.join_path[0].tbl
    right_tbl = jp.join_path[1].tbl

    if left_tbl in ignore_lst or right_tbl in ignore_lst:
        return True
    if left_tbl in excluded_tables or right_tbl in excluded_tables:
        return True
    if left_tbl in size_dic and right_tbl in size_dic:
        if size_dic[left_tbl] > 1_000_000 or size_dic[right_tbl] > 1_000_000:
            return True
    return False

def get_column_lst(joinable_lst, data_dic, size_dic, ignore_lst, path, base_df, class_attr, uninfo):
    """
    Generates a list of JoinColumn objects from a list of joinable paths.
    Applies various filtering rules and loads necessary data.
    """
    new_col_lst = []
    skip_count = 0
    excluded_tables = {"s27g-2w3u.csv", "2013_NYC_School_Survey.csv", "5a8g-vpdd.csv"}
    skip_columns = {"class"}

    for i, jp in enumerate(joinable_lst):
        logging.info(f"Processing join path {i}: {jp}")

        if should_skip_join(jp, ignore_lst, size_dic, excluded_tables):
            skip_count += 1
            continue

        df_l = load_dataframe(jp.join_path[0].tbl, path, data_dic)
        df_r = load_dataframe(jp.join_path[1].tbl, path, data_dic)

        if df_l is None or df_r is None:
            skip_count += 1
            continue

        left_col = jp.join_path[0].col
        right_col = jp.join_path[1].col

        if right_col not in df_r.columns or left_col not in df_l.columns:
            skip_count += 1
            continue

        if df_r.dtypes[right_col] in ["float64", "int64"]:
            skip_count += 1
            continue

        for col in df_r.columns:
            if col == right_col or col in skip_columns or left_col in skip_columns:
                continue

            jc = JoinColumn(jp, df_r, col, base_df, class_attr, len(new_col_lst), uninfo)
            new_col_lst.append(jc)

            if jc.column == "School Type" and jp.join_path[1].tbl == "bnea-fu3k.csv":
                logging.info(f"Key column found at index {len(new_col_lst)-1}: {jc.column}")

    return new_col_lst, skip_count

class JoinPath:
    """
    Represents a sequence of joinable keys between tables.
    """
    def __init__(self, join_key_list):
        self.join_path = join_key_list

    def to_str(self):
        """Returns a string representation of the join path."""
        return " JOIN ".join(f"{key.tbl[:-4]}.{key.col}" for key in self.join_path)

    def set_df(self, data_dic):
        """Assigns DataFrames to each JoinKey in the path."""
        for key in self.join_path:
            key.dataset = data_dic.get(key.tbl, None)

    def print_metadata_str(self):
        """Prints human-readable metadata for each key in the path."""
        print(self.to_str())
        for key in self.join_path:
            print(f"{key.tbl[:-4]}.{key.col}")
            print(f"datasource: {key.tbl}, unique_values: {key.unique_values}, non_empty_values: {key.non_empty}, "
                  f"total_values: {key.total_values}, join_card: {get_join_type(key.join_card)}, "
                  f"jaccard_similarity: {key.js}, jaccard_containment: {key.jc}")

    def get_distance(self, other):
        """Returns a distance between join paths (currently a placeholder)."""
        return 0

    def __str__(self):
        return f"JoinPath({self.to_str()})"

    def __repr__(self):
        return str(self)

class JoinKey:
    """
    Represents a single joinable key from a dataset, with metadata for evaluation.
    """
    def __init__(self, col_drs=None, unique_values=0, total_values=0, non_empty=0):
        self.dataset = ""
        try:
            self.tbl = col_drs.source_name
            self.col = col_drs.field_name
        except AttributeError:
            self.tbl = ""
            self.col = ""

        self.unique_values = unique_values
        self.total_values = total_values
        self.non_empty = non_empty

        try:
            self.join_card = col_drs.metadata.get("join_card", 0)
            self.js = col_drs.metadata.get("js", 0)
            self.jc = col_drs.metadata.get("jc", 0)
        except Exception:
            self.join_card = 0
            self.js = 0
            self.jc = 0

def get_join_type(join_card):
    """
    Converts a join cardinality value to a readable string.
    """
    return {
        0: "One-to-One",
        1: "One-to-Many",
        2: "Many-to-One",
    }.get(join_card, "Many-to-Many")

def find_farthest(distance_dic):
    """
    Finds the key with the maximum distance from the dictionary.
    """
    return max(distance_dic, key=distance_dic.get, default=-1)

def get_clusters(assignment, k):
    """
    Groups assigned join paths into k clusters based on assignment map.
    """
    clusters = [[] for _ in range(k)]
    for key, group_id in assignment.items():
        clusters[group_id].append(key)
    return clusters

def cluster_join_paths(joinable_lst, k, epsilon):
    """
    Clusters join paths using a basic k-center approach with distance threshold.
    """
    random.seed(0)
    centers = []
    assignment = {}
    distance = {}
    max_dist = 0

    for i in range(k):
        if i == 0:
            centers.append(random.randint(0, len(joinable_lst) - 1))
        else:
            centers.append(find_farthest(distance))

        for idx, jp in enumerate(joinable_lst):
            if i == 0:
                assignment[jp] = 0
                distance[idx] = jp.get_distance(joinable_lst[centers[-1]])
            else:
                new_dist = jp.get_distance(joinable_lst[centers[-1]])
                if new_dist < distance.get(idx, float('inf')):
                    assignment[jp] = i
                    distance[idx] = new_dist

        if max(distance.values(), default=0) < epsilon:
            break

    return centers, assignment, get_clusters(assignment, k)

def get_join_paths_from_file(querydata, filepath):
    """
    Extracts join paths from a CSV file where the given table name appears.
    """
    df = pd.read_csv(filepath)
    options = []

    for _, row in df.iterrows():
        jk1 = JoinKey()
        jk2 = JoinKey()
        jk1.tbl = row.get("tbl1", "")
        jk1.col = row.get("col1", "")
        jk2.tbl = row.get("tbl2", "")
        jk2.col = row.get("col2", "")

        if querydata in [jk1.tbl, jk2.tbl]:
            options.append(JoinPath([jk1, jk2] if jk1.tbl == querydata else [jk2, jk1]))

    return options
