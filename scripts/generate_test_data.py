"""
Генерация тестовых данных для интерфейса из данных ноутбуков
Создает JSON файлы с примерами текстов для демонстрации работы веб-интерфейса
"""

import json
import os
from pathlib import Path

# Примеры данных из ноутбуков и analysis_summary.json
# Используем структуру данных из ISOT/Kaggle датасета

test_examples = {
    "fake_news": [
        {
            "id": 1,
            "title": "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",
            "text": "Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and  the very dishonest fake news media.  The former reality show star had just one job to do and he couldn t do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year,  President Angry Pants tweeted.",
            "label": "fake",
            "confidence": 0.92,
            "subject": "News",
            "date": "December 31, 2017"
        },
        {
            "id": 2,
            "title": "Drunk Bragging Trump Staffer Started Russian Collusion Investigation",
            "text": "House Intelligence Committee Chairman Devin Nunes is going to have a bad day. He s been under the assumption, like many of us, that the Christopher Steele-dossier was what prompted the Russia investigation so he s been lashing out at the Department of Justice and the FBI in order to protect Trump. As it happens, the dossier is not what started the investigation, according to documents obtained by the New York Times.",
            "label": "fake",
            "confidence": 0.88,
            "subject": "News",
            "date": "December 31, 2017"
        },
        {
            "id": 3,
            "title": "Sheriff David Clarke Becomes An Internet Joke For Threatening To Poke People 'In The Eye'",
            "text": "On Friday, it was revealed that former Milwaukee Sheriff David Clarke, who was being considered for Homeland Security Secretary in Donald Trump s administration, has an email scandal of his own. I am UNINTIMIDATED by lib media attempts to smear and discredit me with their FAKE NEWS reports designed to silence me,  the former sheriff tweeted.",
            "label": "fake",
            "confidence": 0.85,
            "subject": "News",
            "date": "December 30, 2017"
        },
        {
            "id": 4,
            "title": "Scientists Discover That Eating Chocolate Every Day Helps You Lose Weight",
            "text": "A new study published in the Journal of Nutrition claims that eating chocolate every day can help you lose weight. Researchers found that participants who consumed dark chocolate daily lost an average of 10 pounds in just two weeks without changing their diet or exercise routine. The study was funded by a major chocolate manufacturer.",
            "label": "fake",
            "confidence": 0.90,
            "subject": "Health",
            "date": "January 15, 2024"
        },
        {
            "id": 5,
            "title": "Breaking: Major Tech Company Announces Free Internet for Everyone",
            "text": "In an unprecedented move, a major tech company has announced plans to provide free high-speed internet to every household in the world. The CEO made the announcement during a surprise press conference, stating that the service will be completely free and funded by advertising revenue. No contracts or hidden fees required.",
            "label": "fake",
            "confidence": 0.87,
            "subject": "Technology",
            "date": "February 1, 2024"
        }
    ],
    "real_news": [
        {
            "id": 1,
            "title": "As U.S. budget fight looms, Republicans flip their fiscal script",
            "text": "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a fiscal conservative on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBS's Face the Nation, drew a hard line on federal spending, which lawmakers are bracing to do battle over in January.",
            "label": "real",
            "confidence": 0.94,
            "subject": "politicsNews",
            "date": "December 31, 2017"
        },
        {
            "id": 2,
            "title": "U.S. military to accept transgender recruits on Monday: Pentagon",
            "text": "WASHINGTON (Reuters) - Transgender people will be allowed for the first time to enlist in the U.S. military starting on Monday as ordered by federal courts, the Pentagon said on Friday, after President Donald Trump's administration decided not to appeal rulings that blocked his transgender ban. Two federal appeals courts, one in Washington and one in Virginia, last week rejected the administration's request to put on hold orders by lower court judges requiring the military to begin accepting transgender recruits on Jan. 1.",
            "label": "real",
            "confidence": 0.91,
            "subject": "politicsNews",
            "date": "December 29, 2017"
        },
        {
            "id": 3,
            "title": "Senior U.S. Republican senator: 'Let Mr. Mueller do his job'",
            "text": "WASHINGTON (Reuters) - The special counsel investigation of links between Russia and President Trump's 2016 election campaign should continue without interference in 2018, despite calls from some Trump administration allies and Republican lawmakers to shut it down, a prominent Republican senator said on Sunday. Lindsey Graham, who serves on the Senate armed forces and judiciary committees, said Department of Justice Special Counsel Robert Mueller needs to carry on with his Russia investigation without political interference.",
            "label": "real",
            "confidence": 0.93,
            "subject": "politicsNews",
            "date": "December 31, 2017"
        },
        {
            "id": 4,
            "title": "New Study Shows Benefits of Regular Exercise on Mental Health",
            "text": "A comprehensive study published in the Journal of Clinical Psychology has found that regular exercise significantly improves mental health outcomes. The research, conducted over five years with over 10,000 participants, shows that individuals who engaged in at least 30 minutes of moderate exercise daily reported lower levels of anxiety and depression. The study was peer-reviewed and published in a reputable scientific journal.",
            "label": "real",
            "confidence": 0.89,
            "subject": "Health",
            "date": "January 20, 2024"
        },
        {
            "id": 5,
            "title": "Tech Companies Announce New Privacy Standards",
            "text": "Major technology companies have jointly announced new privacy standards aimed at better protecting user data. The initiative, developed in collaboration with privacy advocates and regulators, includes enhanced encryption, clearer data usage policies, and improved user controls. The standards will be implemented gradually over the next 18 months.",
            "label": "real",
            "confidence": 0.92,
            "subject": "Technology",
            "date": "February 5, 2024"
        }
    ],
    "mixed_examples": [
        {
            "id": 1,
            "text": "Scientists have discovered a new planet that could potentially support life. The planet, located 120 light-years away, has conditions similar to Earth and contains water in its atmosphere.",
            "expected_label": "real",
            "category": "science"
        },
        {
            "id": 2,
            "text": "BREAKING: The government has secretly been hiding evidence of alien contact for decades. Documents leaked by an anonymous source reveal that multiple UFO sightings were covered up.",
            "expected_label": "fake",
            "category": "conspiracy"
        },
        {
            "id": 3,
            "text": "The Federal Reserve announced today that it will maintain current interest rates, citing stable economic indicators. This decision comes after months of speculation about potential rate changes.",
            "expected_label": "real",
            "category": "economics"
        },
        {
            "id": 4,
            "text": "A viral video shows a politician accepting bribes in exchange for votes. The footage, which has been viewed millions of times, allegedly shows clear evidence of corruption.",
            "expected_label": "fake",
            "category": "politics"
        },
        {
            "id": 5,
            "text": "Medical researchers have made significant progress in developing a new treatment for Alzheimer's disease. Clinical trials show promising results, with patients showing improved cognitive function.",
            "expected_label": "real",
            "category": "health"
        }
    ],
    "model_comparison": {
        "models": [
            {
                "name": "LSTM Baseline",
                "accuracy": 0.89,
                "f1_score": 0.88,
                "precision": 0.87,
                "recall": 0.89,
                "inference_time_ms": 45
            },
            {
                "name": "CNN-Text Baseline",
                "accuracy": 0.91,
                "f1_score": 0.90,
                "precision": 0.90,
                "recall": 0.91,
                "inference_time_ms": 38
            },
            {
                "name": "BERT-base-uncased",
                "accuracy": 0.95,
                "f1_score": 0.94,
                "precision": 0.94,
                "recall": 0.95,
                "inference_time_ms": 120
            },
            {
                "name": "DistilBERT",
                "accuracy": 0.94,
                "f1_score": 0.93,
                "precision": 0.93,
                "recall": 0.94,
                "inference_time_ms": 65
            }
        ],
        "best_model": "BERT-base-uncased",
        "fastest_model": "CNN-Text Baseline",
        "balanced_model": "DistilBERT"
    }
}

# Создаем директорию для тестовых данных интерфейса
output_dir = Path("data/interface_test_data")
output_dir.mkdir(parents=True, exist_ok=True)

# Сохраняем полный набор тестовых данных
output_file = output_dir / "test_examples.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(test_examples, f, indent=2, ensure_ascii=False)

print(f"✓ Создан файл: {output_file}")

# Создаем отдельные файлы для разных целей
# 1. Примеры фейковых новостей
fake_file = output_dir / "fake_examples.json"
with open(fake_file, 'w', encoding='utf-8') as f:
    json.dump({"examples": test_examples["fake_news"]}, f, indent=2, ensure_ascii=False)

print(f"✓ Создан файл: {fake_file}")

# 2. Примеры реальных новостей
real_file = output_dir / "real_examples.json"
with open(real_file, 'w', encoding='utf-8') as f:
    json.dump({"examples": test_examples["real_news"]}, f, indent=2, ensure_ascii=False)

print(f"✓ Создан файл: {real_file}")

# 3. Смешанные примеры для тестирования
mixed_file = output_dir / "mixed_examples.json"
with open(mixed_file, 'w', encoding='utf-8') as f:
    json.dump({"examples": test_examples["mixed_examples"]}, f, indent=2, ensure_ascii=False)

print(f"✓ Создан файл: {mixed_file}")

# 4. Данные для сравнения моделей
models_file = output_dir / "model_comparison.json"
with open(models_file, 'w', encoding='utf-8') as f:
    json.dump(test_examples["model_comparison"], f, indent=2, ensure_ascii=False)

print(f"✓ Создан файл: {models_file}")

# 5. Упрощенный формат для быстрого тестирования интерфейса
quick_test = {
    "test_cases": [
        {
            "text": "Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that.",
            "label": "fake"
        },
        {
            "text": "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress called himself a fiscal conservative on Sunday.",
            "label": "real"
        },
        {
            "text": "Scientists discover that eating chocolate every day helps you lose weight without any diet changes.",
            "label": "fake"
        },
        {
            "text": "A comprehensive study published in the Journal of Clinical Psychology has found that regular exercise significantly improves mental health outcomes.",
            "label": "real"
        }
    ]
}

quick_test_file = output_dir / "quick_test.json"
with open(quick_test_file, 'w', encoding='utf-8') as f:
    json.dump(quick_test, f, indent=2, ensure_ascii=False)

print(f"✓ Создан файл: {quick_test_file}")

print("\n" + "=" * 60)
print("ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ ЗАВЕРШЕНА")
print("=" * 60)
print(f"\nВсе файлы сохранены в: {output_dir}")
print("\nСозданные файлы:")
print("  - test_examples.json - полный набор тестовых данных")
print("  - fake_examples.json - примеры фейковых новостей")
print("  - real_examples.json - примеры реальных новостей")
print("  - mixed_examples.json - смешанные примеры для тестирования")
print("  - model_comparison.json - данные для сравнения моделей")
print("  - quick_test.json - упрощенные примеры для быстрого тестирования")

