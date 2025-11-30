"""
Скрипт для преобразования .py файлов в .ipynb формат
Используется для создания Jupyter notebooks из Python скриптов
"""
import json
import re
import sys
from pathlib import Path

def py_to_ipynb(py_file_path, output_path=None):
    """Преобразует Python файл в Jupyter notebook"""
    
    with open(py_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Разбиваем на ячейки по комментариям-разделителям
    # Ищем паттерны типа "# =======" или "# ---"
    cell_pattern = r'(#\s*=+\s*.*?=+\s*|#\s*-+\s*.*?-+\s*)'
    
    # Разбиваем на секции
    sections = re.split(cell_pattern, content)
    
    cells = []
    current_cell = []
    current_type = 'code'
    
    # Обрабатываем каждую секцию
    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        # Проверяем, является ли это markdown комментарием
        if section.strip().startswith('#') and ('=' in section or '-' in section):
            # Сохраняем предыдущую ячейку если есть
            if current_cell:
                cells.append({
                    'cell_type': current_type,
                    'metadata': {},
                    'source': ''.join(current_cell),
                    'outputs': [],
                    'execution_count': None
                })
                current_cell = []
            
            # Создаем markdown ячейку из заголовка
            header_text = section.strip('#').strip('=').strip('-').strip()
            if header_text:
                cells.append({
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': f'## {header_text}\n'
                })
        else:
            # Добавляем код в текущую ячейку
            current_cell.append(section)
    
    # Добавляем последнюю ячейку
    if current_cell:
        cells.append({
            'cell_type': 'code',
            'metadata': {},
            'source': ''.join(current_cell),
            'outputs': [],
            'execution_count': None
        })
    
    # Если не удалось разбить на ячейки, создаем одну большую
    if not cells:
        cells = [{
            'cell_type': 'code',
            'metadata': {},
            'source': content,
            'outputs': [],
            'execution_count': None
        }]
    
    # Создаем структуру notebook
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.8.0'
            },
            'colab': {
                'provenance': []
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    # Сохраняем
    if output_path is None:
        output_path = py_file_path.replace('.py', '.ipynb')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Создан notebook: {output_path}")
    print(f"  Ячеек: {len(cells)}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python convert_py_to_ipynb.py <путь_к_файлу.py>")
        sys.exit(1)
    
    py_file = sys.argv[1]
    py_to_ipynb(py_file)

