from transformers import pipeline

text_generator = pipeline("text-generation", model="gpt2")
prompt = """
Review: "Smooth acceleration" → Sentiment: Positive
Review: "Clunky gearbox" → Sentiment: Negative
Review: "Balanced torque curve" → Sentiment: Positive
Review: "Great brake power" → Sentiment: Positive
Review: "Poor brake response" → Sentiment: Negative
Review: "Good balance" → Sentiment: Positive
Review: "Noisy engine" → Sentiment: Negative
Review: "Excellent feel" → Sentiment: Positive
Review: "Unpredictable handling" → Sentiment: Negative
Review: "Mediocre steering" → Sentiment: Neutral
Review: "Great handling" → Sentiment:
"""
result = text_generator(prompt, max_length=50)
print(result[0]['generated_text'])
