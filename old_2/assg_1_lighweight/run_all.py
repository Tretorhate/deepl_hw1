"""
Barlyk esepterdi oryndau
Iske qosu: python run_all.py
"""
import os
import sys
import torch

print('='*60)
print('Deep Learning - Assignment 1')
print('='*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

os.makedirs('results', exist_ok=True)

print('\n>>> Problem 1 (MLP)...')
exec(open('problem1_mlp.py').read())

print('\n>>> Problem 2 (Optimization)...')
exec(open('problem2_optimization.py').read())

print('\n>>> Problem 3 (CNN)...')
exec(open('problem3_cnn.py').read())

print('\n>>> Bonus (Transfer Learning)...')
exec(open('bonus_transfer_learning.py').read())

print('\n' + '='*60)
print('DONE!')
print('Results: results/ folder')
print('='*60)
