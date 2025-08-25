#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para verificar versoes do Python instaladas
"""
import sys
import subprocess
import os

def check_python_versions():
    """Verifica todas as versoes do Python instaladas"""
    print("=" * 60)
    print("VERIFICACAO DE VERSOES DO PYTHON")
    print("=" * 60)
    
    # Python atual
    print(f"\nPython atual (python):")
    print(f"  Executavel: {sys.executable}")
    print(f"  Versao: {sys.version}")
    print(f"  Path: {sys.path[0]}")
    
    # Verificar python3
    try:
        result = subprocess.run(['python3', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"\nPython 3 (python3):")
            print(f"  Versao: {result.stdout.strip()}")
        else:
            print(f"\nPython 3 (python3): Nao encontrado")
    except:
        print(f"\nPython 3 (python3): Nao encontrado")
    
    # Verificar py launcher
    try:
        result = subprocess.run(['py', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"\nPython Launcher (py):")
            print(f"  Versao: {result.stdout.strip()}")
        else:
            print(f"\nPython Launcher (py): Nao encontrado")
    except:
        print(f"\nPython Launcher (py): Nao encontrado")
    
    # Verificar pip
    try:
        result = subprocess.run(['pip', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"\nPip atual:")
            print(f"  {result.stdout.strip()}")
        else:
            print(f"\nPip atual: Nao encontrado")
    except:
        print(f"\nPip atual: Nao encontrado")
    
    # Verificar pip3
    try:
        result = subprocess.run(['pip3', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"\nPip 3:")
            print(f"  {result.stdout.strip()}")
        else:
            print(f"\nPip 3: Nao encontrado")
    except:
        print(f"\nPip 3: Nao encontrado")
    
    print("\n" + "=" * 60)
    print("RECOMENDACOES:")
    print("=" * 60)
    
    if sys.version_info[0] == 2:
        print("❌ Python 2.7 detectado - ATUALIZE PARA PYTHON 3!")
        print("\nPara usar Python 3:")
        print("1. python3 run_demo.py")
        print("2. py -3 run_demo.py")
        print("3. Configure o PATH para priorizar Python 3")
    else:
        print("✅ Python 3 detectado - TUDO OK!")
        print("\nExecute:")
        print("python run_demo.py")

if __name__ == "__main__":
    check_python_versions()

