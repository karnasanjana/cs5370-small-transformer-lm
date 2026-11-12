# generate.py - load a trained model and generate text. No training needed.
#
# Examples:
#   python generate.py --model pretrained --prompt "Notice to all students."
#   python generate.py --model finetuned --question "How much is the late fee?"
#   python generate.py --model finetuned            (interactive: type questions)
#   python generate.py --model pretrained           (interactive: type prompts)

import argparse

from slm import generate, load_model, split_words

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["pretrained", "finetuned"], default="finetuned")
parser.add_argument("--prompt", default=None, help="text that the model should continue")
parser.add_argument("--question", default=None, help="question for the model to answer")
parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
parser.add_argument("--max_tokens", type=int, default=50)
args = parser.parse_args()

model, tokenizer = load_model(f"model/{args.model}.pt")
print(f"loaded model/{args.model}.pt  (vocabulary: {len(tokenizer.vocab)} tokens)")


def run(text, is_question):
    # tell the user about words the model has never seen
    for w in split_words(text):
        if w not in tokenizer.word_to_id:
            print(f"  (note: '{w}' is not in the vocabulary and becomes <unk>)")

    if is_question:
        prompt = "question: " + text + " answer:"
    else:
        prompt = text

    output = generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
    print("output:", output)
    print()


if args.prompt is not None:
    run(args.prompt, is_question=False)
elif args.question is not None:
    run(args.question, is_question=True)
else:
    # interactive mode: the fine-tuned model expects questions, the pretrained model free text
    is_question = args.model == "finetuned"
    label = "question" if is_question else "prompt"
    print(f"Type a {label} and press Enter. Empty line to quit.")
    while True:
        text = input(f"{label}> ").strip()
        if text == "":
            break
        run(text, is_question)
