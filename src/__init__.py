import os

# Create __init__.py files
init_files = [
    'src/__init__.py',
    'scripts/__init__.py',
    'tests/__init__.py',
    'data/__init__.py',
    'notebooks/__init__.py'
]

for file in init_files:
    with open(file, 'w') as f:
        f.write('# Package initialization file\n')
    print(f"✓ Created: {file}")