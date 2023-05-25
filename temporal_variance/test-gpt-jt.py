#!/usr/bin/env python
# coding: utf-8

# In[ ]:


prompt = '''
Generate summary of the text given below:

text: 'Once upon a time, there lived a rabbit and tortoise. The rabbit could run fast. He was very proud of his speed. While the turtle was slow and consistent. One day that tortoise came to meet him. The tortoise was walking very slow as usual. The rabbit looked and laughed at him. The tortoise asked “what happened?” The rabbit replied, “You walk so slowly! How can you survive like this?”. The turtle listened to everything and felt humiliated by the rabbit’s words. The tortoise replied, “Hey friend! You are very proud of your speed. Let’s have a race and see who is faster”. The rabbit was surprised by the challenge of the tortoise. But he accepted the challenge as he thought it would be a cakewalk for him. So, the tortoise and rabbit started the race. The rabbit was as usual very fast and went far away. While the tortoise was left behind. After a while, the rabbit looked behind. He said to himself, “The slow turtle will take ages to come near me. I should rest a bit”. The rabbit was tired from running fast. The sun was high too. He ate some grass and decided to take a nap. He said to himself, “I am confident; I can win even if the tortoise passes me. I should rest a bit”. With that thought, he slept and lost the track of time. Meanwhile, the slow and steady turtle kept on moving. Although he was tired, he didn’t rest. Sometime later, he passed the rabbit when the rabbit was still sleeping. The rabbit suddenly woke up after sleeping for a long time. He saw that the tortoise was about to cross the finishing line. He started running very fast with his full energy. But it was too late. The slow turtle had already touched the finishing line. He has already won the race. The rabbit was very disappointed with himself while the tortoise was very happy to win the race with his slow speed. He could not believe his eyes. He was shocked by the end results. At last, the tortoise asked the rabbit “Now who is faster”. The rabbit had learned his lesson. He could not utter a word. The tortoise said bye to the rabbit and left that place calmly and happily.'
'''

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)

