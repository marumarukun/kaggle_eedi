import polars as pl
from sklearn.preprocessing import LabelEncoder

INPUT_DIR = "data/input"


def load_data():
    # データ読み込み
    train = pl.read_csv(f"../{INPUT_DIR}/train.csv", try_parse_dates=True)
    test = pl.read_csv(f"../{INPUT_DIR}/test.csv", try_parse_dates=True)
    return train, test


def add_subject_name_info(train_df: pl.DataFrame, mapping_df: pl.DataFrame) -> pl.DataFrame:
    sub_a_df = (
        train_df.group_by("MisconceptionAId")
        .agg(pl.col("SubjectName").unique().alias("SubjectName_A"))
        .sort("MisconceptionAId")
        .rename({"MisconceptionAId": "MisconceptionId"})
        .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
    )

    sub_b_df = (
        train_df.group_by("MisconceptionBId")
        .agg(pl.col("SubjectName").unique().alias("SubjectName_B"))
        .sort("MisconceptionBId")
        .rename({"MisconceptionBId": "MisconceptionId"})
        .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
    )
    sub_c_df = (
        train_df.group_by("MisconceptionCId")
        .agg(pl.col("SubjectName").unique().alias("SubjectName_C"))
        .sort("MisconceptionCId")
        .rename({"MisconceptionCId": "MisconceptionId"})
        .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
    )
    sub_d_df = (
        train_df.group_by("MisconceptionDId")
        .agg(pl.col("SubjectName").unique().alias("SubjectName_D"))
        .sort("MisconceptionDId")
        .rename({"MisconceptionDId": "MisconceptionId"})
        .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
    )

    mis_id_series = mapping_df.select("MisconceptionId")
    mis_id_and_subject_name_df = (
        mis_id_series.join(sub_a_df, on="MisconceptionId", how="left")
        .join(sub_b_df, on="MisconceptionId", how="left")
        .join(sub_c_df, on="MisconceptionId", how="left")
        .join(sub_d_df, on="MisconceptionId", how="left")
        .with_columns(
            pl.all().exclude("MisconceptionId").fill_null([]),
        )
        .with_columns(pl.col("SubjectName_A").list.concat("SubjectName_B").alias("SubjectNames"))
        .with_columns(pl.col("SubjectNames").list.concat("SubjectName_C").alias("SubjectNames"))
        .with_columns(pl.col("SubjectNames").list.concat("SubjectName_D").alias("SubjectNames"))
        .with_columns(pl.col("SubjectNames").list.unique())
        .select("MisconceptionId", "SubjectNames")
    )
    # misconception_dfと結合

    mapping_meta_df = mapping_df.join(mis_id_and_subject_name_df, on="MisconceptionId", how="left")

    # mappingデータの整形
    mapping_meta_df = mapping_meta_df.with_columns(
        [
            # 複数のSubjectNamesがある場合は、それらを列挙した文字列を生成
            pl.when(pl.col("SubjectNames").list.len() > 0)
            .then(
                pl.lit("The misconception '")
                + pl.col("MisconceptionName")
                + pl.lit("' is primarily observed in the following subjects: ")
                + pl.col("SubjectNames").list.join(", ")
            )
            # 複数のSubjectNamesがない場合は、MisconceptionNameをそのまま使う
            .otherwise(pl.lit("The misconception is: ") + pl.col("MisconceptionName"))
            .alias("MisconceptionName_with_SubjectNames")
        ]
    )
    return mapping_meta_df


# TODO：train or testの整形関数実装
# def preprocess_train_or_test
