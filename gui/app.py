"""Core Tkinter GUI: upload/display images and show CNN predictions."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from shriyans_cnn.predict import DigitPredictor


class HNRSApp(tk.Tk):
    def __init__(self, model_path: Path) -> None:
        super().__init__()
        self.title("Handwritten Number Recognition System")
        self.geometry("820x580")
        self.minsize(720, 520)
        self.configure(bg="#f4f6fb")
        self.model_path = model_path
        self.predictor = None
        self.selected_files: list[str] = []
        self.preview = None
        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), background="#f4f6fb")
        style.configure("Result.TLabel", font=("Segoe UI", 24, "bold"), foreground="#174ea6")
        ttk.Label(self, text="Handwritten Number Recognition", style="Title.TLabel").pack(pady=(24, 6))
        ttk.Label(self, text="CNN model - MNIST - confidence-aware output", background="#f4f6fb").pack()
        body = ttk.Frame(self, padding=24)
        body.pack(fill="both", expand=True, padx=28, pady=20)
        left = ttk.LabelFrame(body, text="1. Input image", padding=18)
        right = ttk.LabelFrame(body, text="2. Prediction", padding=18)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))
        self.image_label = ttk.Label(left, text="Choose one digit image or several pre-segmented digits", anchor="center")
        self.image_label.pack(fill="both", expand=True)
        ttk.Button(left, text="Upload image(s)", command=self.choose_images).pack(fill="x", pady=(14, 0))
        self.result_label = ttk.Label(right, text="No prediction yet", style="Result.TLabel", anchor="center")
        self.result_label.pack(fill="x", pady=(45, 12))
        self.confidence_label = ttk.Label(right, text="", anchor="center")
        self.confidence_label.pack(fill="x")
        self.status_label = ttk.Label(right, text=f"Model: {self.model_path}", wraplength=300, anchor="center")
        self.status_label.pack(fill="x", pady=(35, 12))
        ttk.Button(right, text="Recognise", command=self.recognise).pack(fill="x")

    def choose_images(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select digit images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if not files:
            return
        self.selected_files = list(files)
        image = Image.open(files[0]).convert("RGB")
        image.thumbnail((300, 260))
        self.preview = ImageTk.PhotoImage(image)
        suffix = f"\n+ {len(files)-1} more image(s)" if len(files) > 1 else ""
        self.image_label.configure(image=self.preview, text=suffix, compound="top")

    def recognise(self) -> None:
        if not self.selected_files:
            messagebox.showwarning("No input", "Upload at least one image first.")
            return
        try:
            if self.predictor is None:
                self.predictor = DigitPredictor(self.model_path)
            predictions = [self.predictor.predict(path) for path in self.selected_files]
            number = "".join(str(item[0]) for item in predictions)
            minimum = min(item[1] for item in predictions)
            average = sum(item[1] for item in predictions) / len(predictions)
            self.result_label.configure(text=f"Prediction: {number}")
            self.confidence_label.configure(text=f"Average confidence: {average:.1%}\nLowest digit: {minimum:.1%}")
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/cnn_mnist.keras"))
    args = parser.parse_args()
    HNRSApp(args.model).mainloop()


if __name__ == "__main__":
    main()

