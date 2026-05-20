




def augment_dataset(input_dir, output_dir, num_workers, calculator_path, augmentations):

    for augmentation in augmentations:
        dataset.apply_augmentation(augmentation)
    
    # TODO: relabel with new calculator
    dataset.relabel_with_new_calculator(calculator_path)

    # TODO: save dataset
    dataset.save(output_dir)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--calculator-path", type=str, required=True) # foundation model calculator path like uma
    args = parser.parse_args()
    

    # TODO: also make this scalable by support multiple augmentations
    augment_dataset(
    input_dir=args.input_dir, 
    output_dir=args.output_dir, 
    num_workers=args.num_workers, 
    calculator_path=args.calculator_path,
    augmentations=["rattle", "distort"],
    )

if __name__ == "__main__":
    main()