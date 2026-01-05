"""
Main script to run all experiments for the Deep Learning project.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from section1.train_mlp import run_section1_experiments
from section2.optimizers import run_section2_experiments
from section3.cnn_layers import run_section3_experiments
from section4.resnet import run_section4_experiments


def show_menu():
    """Display interactive menu for section selection."""
    print("=" * 50)
    print("Deep Learning Project - Experiments")
    print("=" * 50)
    print()
    print("Select which section(s) to run:")
    print()
    print("  1. Section 1: MLP and Activations")
    print("  2. Section 2: Optimization")
    print("  3. Section 3: CNNs")
    print("  4. Section 4: Modern Architectures")
    print("  5. Run all sections")
    print("  0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == '0':
                print("Exiting...")
                return None
            elif choice == '1':
                return {'section1': True}
            elif choice == '2':
                return {'section2': True}
            elif choice == '3':
                return {'section3': True}
            elif choice == '4':
                return {'section4': True}
            elif choice == '5':
                return {'all': True}
            else:
                print("Invalid choice. Please enter a number between 0-5.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None
        except Exception as e:
            print(f"Error: {e}. Please try again.")


def main():
    parser = argparse.ArgumentParser(
        description='Deep Learning Project - Run Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Interactive menu
  python main.py 1            # Run Section 1
  python main.py --section1   # Run Section 1
  python main.py --all        # Run all sections
        """
    )
    parser.add_argument('section', nargs='?', type=int, 
                       help='Section number to run (1-4), or use --all for all sections')
    parser.add_argument('--all', action='store_true', help='Run all sections')
    parser.add_argument('--section1', action='store_true', help='Run Section 1: MLP')
    parser.add_argument('--section2', action='store_true', help='Run Section 2: Optimization')
    parser.add_argument('--section3', action='store_true', help='Run Section 3: CNNs')
    parser.add_argument('--section4', action='store_true', help='Run Section 4: Modern Architectures')
    
    args = parser.parse_args()
    
    # Determine which sections to run
    run_section1 = False
    run_section2 = False
    run_section3 = False
    run_section4 = False
    
    # Check command-line arguments
    if args.all:
        run_section1 = run_section2 = run_section3 = run_section4 = True
    elif args.section1:
        run_section1 = True
    elif args.section2:
        run_section2 = True
    elif args.section3:
        run_section3 = True
    elif args.section4:
        run_section4 = True
    elif args.section is not None:
        # Positional argument (e.g., python main.py 1)
        if args.section == 1:
            run_section1 = True
        elif args.section == 2:
            run_section2 = True
        elif args.section == 3:
            run_section3 = True
        elif args.section == 4:
            run_section4 = True
        elif args.section == 0:
            # Run all (alternative to --all)
            run_section1 = run_section2 = run_section3 = run_section4 = True
        else:
            print(f"Invalid section number: {args.section}. Please use 1-4, or 0 for all.")
            return
    else:
        # No arguments provided - show interactive menu
        menu_choice = show_menu()
        if menu_choice is None:
            return
        
        if menu_choice.get('all'):
            run_section1 = run_section2 = run_section3 = run_section4 = True
        elif menu_choice.get('section1'):
            run_section1 = True
        elif menu_choice.get('section2'):
            run_section2 = True
        elif menu_choice.get('section3'):
            run_section3 = True
        elif menu_choice.get('section4'):
            run_section4 = True
    
    # Run selected sections
    print("=" * 50)
    print("Deep Learning Project - Experiments")
    print("=" * 50)
    print()
    
    if run_section1:
        print("Running Section 1: MLP and Activations...")
        print("-" * 50)
        run_section1_experiments()
        print()
    
    if run_section2:
        print("Running Section 2: Optimization...")
        print("-" * 50)
        run_section2_experiments()
        print()
    
    if run_section3:
        print("Running Section 3: CNNs...")
        print("-" * 50)
        run_section3_experiments()
        print()
    
    if run_section4:
        print("Running Section 4: Modern Architectures...")
        print("-" * 50)
        run_section4_experiments()
        print()
    
    print("=" * 50)
    print("All experiments completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
