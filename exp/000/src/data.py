import polars as pl

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
            # SubjectNamesがある場合は、それらを列挙した文字列を生成
            pl.when(pl.col("SubjectNames").list.len() > 0)
            .then(
                pl.lit("The misconception '")
                + pl.col("MisconceptionName")
                + pl.lit("' is primarily observed in the following subjects: ")
                + pl.col("SubjectNames").list.join(", ")
            )
            # SubjectNamesがない場合は、MisconceptionNameをそのまま使う
            .otherwise(pl.lit("The misconception is: ") + pl.col("MisconceptionName"))
            .alias("MisconceptionName_with_SubjectNames")
        ]
    )
    return mapping_meta_df


# trainの前処理関数実装
def preprocess_train(train_df: pl.DataFrame) -> pl.DataFrame:
    common_col = [
        "QuestionId",
        "ConstructName",
        "SubjectName",
        "QuestionText",
        "CorrectAnswer",
    ]

    train_long = (
        train_df.select(pl.col(common_col + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]]))
        .unpivot(
            index=common_col,
            variable_name="AnswerType",
            value_name="AnswerText",
        )
        .with_columns(
            (
                pl.lit("ConstructName: ")
                + pl.col("ConstructName")
                + pl.lit(" SubjectName: ")
                + pl.col("SubjectName")
                + pl.lit(" QuestionText: ")
                + pl.col("QuestionText")
                + pl.lit(" AnswerText: ")
                + pl.col("AnswerText")
            ).alias("AllText"),
            pl.col("AnswerType").str.extract(r"Answer([A-D])Text$").alias("AnswerAlphabet"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("QuestionId"),
                    pl.col("AnswerAlphabet"),
                ],
                separator="_",
            ).alias("QuestionId_Answer"),
        )
        .sort("QuestionId_Answer")
    )
    train_misconception_long = (
        train_df.select(pl.col(common_col + [f"Misconception{alpha}Id" for alpha in ["A", "B", "C", "D"]]))
        .unpivot(
            index=common_col,
            variable_name="MisconceptionType",
            value_name="MisconceptionId",
        )
        .with_columns(
            pl.col("MisconceptionType").str.extract(r"Misconception([A-D])Id$").alias("AnswerAlphabet"),
        )
        .with_columns(
            pl.concat_str([pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_").alias("QuestionId_Answer"),
        )
        .sort("QuestionId_Answer")
        .select(pl.col(["QuestionId_Answer", "MisconceptionId"]))
        .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
    )
    # join MisconceptionId
    train_long = train_long.join(train_misconception_long, on="QuestionId_Answer")

    # CorrectAnswerとAnswerAlphabetが一致するもの（つまり正解）は除外
    # また、MisconceptionIdがNoneのものも除外
    train_long = train_long.filter(pl.col("CorrectAnswer") != pl.col("AnswerAlphabet")).drop_nulls(
        subset=["MisconceptionId"]
    )

    return train_long
