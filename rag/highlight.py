from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "zilliz/semantic-highlight-bilingual-v1",
    trust_remote_code=True
)

# 切换到 CPU 以避免 MPS 内存问题
model = model.to('cpu')

# 测试用例
test_cases = [
    {
        "question": "What are the symptoms of dehydration?",
        "context": """
Dehydration occurs when your body loses more fluid than you take in.
Common signs include feeling thirsty and having a dry mouth.
The human body is composed of about 60% water.
Dark yellow urine and infrequent urination are warning signs.
Water is essential for many bodily functions.
Dizziness, fatigue, and headaches can indicate severe dehydration.
Drinking 8 glasses of water daily is often recommended.
"""
    },
    {
        "question": "How does climate change affect polar bears?",
        "context": """
Polar bears are native to the Arctic region.
Climate change is causing Arctic ice to melt at unprecedented rates.
Polar bears depend on sea ice for hunting seals, their primary food source.
Many polar bear populations have been declining in recent years.
The Arctic ecosystem is complex and interconnected.
As ice disappears earlier each spring, bears have less time to hunt.
Some bears are forced to swim longer distances between ice floes.
"""
    },
    {
        "question": "什么是机器学习？",
        "context": """
机器学习是人工智能的一个分支。
它使计算机能够在没有明确编程的情况下学习。
深度学习是机器学习的一个子领域。
机器学习算法通过数据训练来改进性能。
人工智能在医疗、金融等领域有广泛应用。
神经网络是一种常见的机器学习模型。
数据质量对机器学习模型的效果至关重要。
"""
    },
    {
        "question": "What are the benefits of regular exercise?",
        "context": """
Exercise is any bodily activity that enhances physical fitness.
Regular physical activity can improve cardiovascular health and reduce heart disease risk.
Many people enjoy outdoor activities like hiking and cycling.
Exercise helps maintain healthy weight and improves metabolism.
The gym industry has grown significantly in recent years.
Regular workouts can boost mood and reduce symptoms of depression and anxiety.
Protein is important for muscle recovery after exercise.
"""
    },
    {
        "question": "How do vaccines work?",
        "context": """
Vaccines are biological preparations that provide immunity to diseases.
They work by stimulating the immune system to recognize and fight pathogens.
The first vaccine was developed by Edward Jenner in 1796.
Vaccines contain weakened or inactive parts of a pathogen.
Many childhood diseases have been nearly eliminated through vaccination programs.
When vaccinated, the body produces antibodies without getting sick.
Public health campaigns promote vaccination to prevent disease outbreaks.
"""
    }
]

import time

print("=" * 80)
print("连续推理性能测试")
print("=" * 80)

total_start = time.time()

for idx, test_case in enumerate(test_cases, 1):
    question = test_case["question"]
    context = test_case["context"]
    
    print(f"\n{'=' * 80}")
    print(f"测试 {idx}/{len(test_cases)}")
    print(f"问题: {question}")
    print(f"{'-' * 80}")
    
    start = time.time()
    result = model.process(
        question=question,
        context=context,
        threshold=0.5,
        return_sentence_metrics=True,
    )
    elapsed = time.time() - start
    
    highlighted = result["highlighted_sentences"]
    total_sentences = len(context.strip().split('.'))-1 if '。' not in context else len(context.strip().split('。'))-1
    
    print(f"\n✅ 推理耗时: {elapsed:.2f}s")
    print(f"📊 高亮句子: {len(highlighted)}/{total_sentences}")
    print(f"\n高亮内容:")
    for i, sent in enumerate(highlighted, 1):
        print(f"  {i}. {sent.strip()}")
    
    if "sentence_probabilities" in result:
        probs = result["sentence_probabilities"]
        print(f"\n句子概率: {[f'{p:.3f}' for p in probs]}")

total_elapsed = time.time() - total_start
avg_time = total_elapsed / len(test_cases)

print(f"\n{'=' * 80}")
print(f"性能总结")
print(f"{'=' * 80}")
print(f"总测试用例: {len(test_cases)}")
print(f"总耗时: {total_elapsed:.2f}s")
print(f"平均耗时: {avg_time:.2f}s/query")
print(f"吞吐量: {len(test_cases)/total_elapsed:.2f} queries/s")
print(f"{'=' * 80}")
