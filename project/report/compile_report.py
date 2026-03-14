"""
compile_report.py
编译 report.typ → report.pdf

用法：
  cd project/report
  python compile_report.py

依赖：
  pip install typst matplotlib
  sudo apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei
"""

import os
import sys

# ── 确保工作目录在 project/report ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

# ── 生成图表 ──────────────────────────────────────────────────────────────────
print("Step 1/2: Generating figures...")
import subprocess
result = subprocess.run(
    [sys.executable, 'generate_figures.py'],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("Figure generation error:")
    print(result.stderr)
    sys.exit(1)
print(result.stdout.strip())

# ── 编译 Typst 报告 ──────────────────────────────────────────────────────────
print("\nStep 2/2: Compiling Typst report...")
import typst

# CJK 字体搜索路径
font_dirs = [
    '/usr/share/fonts/truetype/wqy/',
    '/usr/share/fonts/opentype/noto/',
    '/usr/share/fonts/truetype/noto/',
    '/Library/Fonts/',
    os.path.expanduser('~/Library/Fonts/'),
    'C:/Windows/Fonts',
]
existing_font_dirs = [d for d in font_dirs if os.path.isdir(d)]

try:
    pdf_bytes, warnings = typst.compile_with_warnings(
        'report.typ',
        output='report.pdf',
        root='.',
        font_paths=existing_font_dirs,
    )
    size = os.path.getsize('report.pdf')
    if warnings:
        for w in warnings:
            msg = str(w)
            if 'unknown font family' not in msg:
                print(f"  Warning: {msg}")
    print(f"\n✓ report.pdf generated successfully ({size:,} bytes)")
    print(f"  Location: {os.path.abspath('report.pdf')}")
except Exception as e:
    print(f"Compilation error: {e}")
    sys.exit(1)
