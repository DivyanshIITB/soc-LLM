from chatbot import get_answer_from_pdf
from difflib import SequenceMatcher

eval_data = [
    {
        "question": "How many credits are needed for a minor?",
        "expected": "30 credits."
    },
    {
        "question": "Can a dual degree student apply for a minor?",
        "expected": "Yes."
    },
    {
        "question": "What happens if you don't complete 30 credits for a minor?",
        "expected": "If you don't complete 30 credits for a minor, the minor will not be awarded. However, the individual course credits will still reflect in the transcript."
    }
]

def is_similar(a, b, threshold=0.6):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

correct = 0

for item in eval_data:
    pred = get_answer_from_pdf(item["question"])
    print(f"\nQ: {item['question']}")
    print(f"✅ Expected: {item['expected']}")
    print(f"🤖 Predicted: {pred}")
    if is_similar(item["expected"], pred):
        correct += 1

accuracy = correct / len(eval_data)
print(f"\n📊 Manual Accuracy: {accuracy * 100:.2f}%")
