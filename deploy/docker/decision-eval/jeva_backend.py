"""Translate the existing evaluation contract to the official Jeva SDK."""


def predict_jeva(runtime, body):
    from jeva import Choice, Noul, Score

    questions = {}
    score_labels = {}
    for name, question in body["questions"].items():
        instructions = question.get("instructions", "")
        if question["type"] == "choice":
            questions[name] = Choice(instructions, question["criteria"], allow_null=False)
        elif question["type"] == "noul":
            questions[name] = Noul(instructions)
        elif question["type"] == "score":
            criteria = question["criteria"]
            questions[name] = Score(instructions, criteria, values=list(range(len(criteria))))
            score_labels[name] = criteria
        else:
            raise ValueError("Unsupported Jeva question type")

    response = runtime.system_one(state=body["state"], questions=questions)
    if set(response.answers) != set(questions):
        raise RuntimeError("Jeva returned incomplete answers")
    answers = {}
    for name, native in response.answers.items():
        probabilities = native.probabilities
        if name in score_labels:
            probabilities = {str(i): probabilities[label] for i, label in enumerate(score_labels[name])}
        answer = {"type": native.kind, "probabilities": probabilities, "confidence": native.confidence}
        answer[native.kind] = getattr(native, native.kind)
        answers[name] = answer
    usage = response.usage.to_dict()
    if usage.get("truncated_tokens", 0):
        raise RuntimeError("Jeva unexpectedly truncated the request")
    return {
        "model": body["model"], "answers": answers, "usage": usage,
        "execution": {"runtime": "jeva", "cross_request_batching": False,
                      "cache_mode": runtime.cache, "option_chunk_size": runtime.option_chunk_size,
                      "option_batch_tokens": runtime.option_batch_tokens},
    }
