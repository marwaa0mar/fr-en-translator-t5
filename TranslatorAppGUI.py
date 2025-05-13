import tkinter as tk
from tkinter import messagebox
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from pathlib import Path

# Load tokenizer and model
model_path = Path(r"D:\Downloads\results\t5-fr-en-translation-best")

tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
model.eval()

# Actual translation function using the loaded model
def translate_with_model(text):
    prefix = "translate English to French: "
    input_text = prefix + text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    print("Input IDs:", input_ids)  # Debugging line
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)
    print("Output IDs:", output_ids)  # Debugging line
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Decoded Output: {decoded_output}")
    return decoded_output

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()

    def setup_ui(self):
        self.root.title("French to English Translator")
        self.root.geometry("700x500")
        self.root.resizable(False, False)
        self.root.configure(bg="#f5f5f5")

        self.create_language_labels()
        self.create_text_areas()
        self.create_french_keyboard()
        self.create_action_buttons()

    def create_language_labels(self):
        frame = tk.Frame(self.root, bg="#f5f5f5")
        frame.pack(pady=10)

        tk.Label(frame, text="French", font=("Arial", 12), bg="#f5f5f5").grid(row=0, column=0, padx=140)
        tk.Label(frame, text="English", font=("Arial", 12), bg="#f5f5f5").grid(row=0, column=1, padx=140)

    def create_text_areas(self):
        frame = tk.Frame(self.root, bg="#f5f5f5")
        frame.pack(pady=10)

        self.input_text = tk.Text(frame, wrap="word", width=30, height=6, font=("Arial", 12))
        self.input_text.grid(row=0, column=0, padx=10, pady=10)

        self.output_text = tk.Text(frame, wrap="word", width=30, height=6, font=("Arial", 12), state="disabled")
        self.output_text.grid(row=0, column=1, padx=10, pady=10)

    def create_french_keyboard(self):
        special_chars = ['é', 'è', 'ê', 'ë', 'à', 'â', 'î', 'ï', 'ô', 'ù', 'û', 'ç', 'œ', 'æ', 'É', 'È', 'Ç', '«', '»']
        frame = tk.Frame(self.root, bg="#f5f5f5")
        frame.pack(pady=5)

        for idx, char in enumerate(special_chars):
            btn = tk.Button(frame, text=char, width=3, font=("Arial", 12),
                            command=lambda c=char: self.insert_char(c))
            btn.grid(row=0, column=idx, padx=2)

    def insert_char(self, char):
        self.input_text.insert(tk.INSERT, char)

    def create_action_buttons(self):
        frame = tk.Frame(self.root, bg="#f5f5f5")
        frame.pack(pady=20)

        translate_btn = tk.Button(frame, text="Translate", command=self.translate, font=("Arial", 12), bg="#4CAF50", fg="white", width=15)
        translate_btn.grid(row=0, column=0, padx=10)

        reset_btn = tk.Button(frame, text="Reset", command=self.reset, font=("Arial", 12), bg="#f44336", fg="white", width=15)
        reset_btn.grid(row=0, column=1, padx=10)

    def translate(self):
        try:
            text = self.input_text.get("1.0", tk.END).strip()

            if not text:
                return messagebox.showwarning("Warning", "Please enter text to translate.")

            translated = translate_with_model(text)
            self.show_output(translated)

        except Exception as e:
            self.show_error("translation", e)

    def reset(self):
        try:
            self.input_text.delete("1.0", tk.END)
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.config(state="disabled")
        except Exception as e:
            self.show_error("resetting", e)

    def show_output(self, text):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state="disabled")

    def show_error(self, action, error):
        messagebox.showerror("Error", f"An error occurred while {action}.\n\nDetails:\n{str(error)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
