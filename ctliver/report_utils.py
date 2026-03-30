def generate_latex_table(report_dict, path="results/summary.tex"):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\textbf{Class} & Precision & Recall & F1-Score \\\\\n\\hline\n")
        for key in report_dict.keys():
            if key in ["0", "1"]:   # only numeric classes
                cls = report_dict[key]
                f.write(f"{key} & {cls['precision']:.2f} & {cls['recall']:.2f} & {cls['f1-score']:.2f} \\\\\n")
        
        # Add accuracy row
        if "accuracy" in report_dict:
            acc = report_dict["accuracy"]
            if isinstance(acc, dict):  # Some sklearn versions return a dict
                f.write(f"accuracy & {acc['precision']:.2f} & {acc['recall']:.2f} & {acc['f1-score']:.2f} \\\\\n")
            else:  # Direct float value
                f.write(f"accuracy & {acc:.2f} & {acc:.2f} & {acc:.2f} \\\\\n")
        
        f.write("\\end{tabular}")
