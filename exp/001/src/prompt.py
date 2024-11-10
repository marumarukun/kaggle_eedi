import re

import polars as pl

# プロンプト
prompt = """
Question Details:
Subject: {SubjectName}
Topic: {ConstructName}
Question: {Question}
Correct Answer: {CorrectAnswer}
Student's Incorrect Answer: {IncorrectAnswer}

You are an experienced mathematics teacher analyzing student misconceptions. Your task is to identify the underlying misconceptions that led to this incorrect answer.

Below are {k} potential misconceptions identified by semantic similarity analysis for this specific question:
{misconception_topk}

Instructions:
1. From these semantically similar misconceptions, select 25 that are most likely to explain this student's error
2. Rank your selections by confidence level (most likely first)
3. Provide only the numbers in a comma-separated format (e.g., 1,10,11,12,...)

Key considerations:
- Consider the student's likely problem-solving process
- Take into account how well each misconception matches the specific error pattern
- Pay special attention to the semantic relevance already identified
- Consider how the incorrect answer might have been derived from these misconceptions

Output format: [numbers only, comma-separated]
"""


# プロンプトを用いてテキストを前処理する用の関数
def preprocess_text(x):
    x = re.sub(r"http\w+", "", x)  # Delete URL
    x = re.sub(r"\.+", ".", x)  # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = x.strip()  # Remove empty characters at the beginning and end
    return x


def apply_template(row, tokenizer, k):
    messages = [
        {
            "role": "user",
            "content": preprocess_text(
                prompt.format(
                    ConstructName=row["ConstructName"],
                    SubjectName=row["SubjectName"],
                    Question=row["QuestionText"],
                    IncorrectAnswer=row["AnswerText"],
                    CorrectAnswer=row["CorrectAnswerText"],
                    k=k,
                    misconception_topk=row["misconception_topk"],
                )
            ),
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def create_prompt(topk_ids, mapping_meta_df, test_long, tokenizer, k):
    # まずStage1で取得したtopkの結果を、misconception_topk列としてtest_longに追加
    misconception_topk_list = []
    for topk_id in topk_ids:
        misconception_list = []
        for i, mid in enumerate(topk_id):
            misconception_list.append(f"{i+1}. {mapping_meta_df['MisconceptionName'][mid['corpus_id']]}")
        misconception_topk = "\n".join(misconception_list)
        misconception_topk_list.append(misconception_topk)

    test_long = test_long.with_columns(pl.Series(misconception_topk_list).alias("misconception_topk"))

    # apply_templateを適用し、prompt列を作成
    test_long = test_long.with_columns(
        pl.Series(
            name="prompt",
            values=[apply_template(row, tokenizer, k) for row in test_long.iter_rows(named=True)],
        )
    )
    return test_long
