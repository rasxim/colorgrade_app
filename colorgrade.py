import tkinter as tk
from tkinter import filedialog, messagebox
import cv2 as cv
from PIL import Image, ImageTk
import numpy as np

class ColorCorrectorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Image Color Corrector")

        self.image_path = None
        self.original_image = None
        self.corrected_image = None

        self.setup_ui()

    def setup_ui(self):
        # Canvas to display images
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="gray")
        self.canvas.pack()

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_image, width=15)
        load_btn.grid(row=0, column=0, padx=5)

        correct_btn = tk.Button(btn_frame, text="Color Correct", command=self.color_correct, width=15)
        correct_btn.grid(row=0, column=1, padx=5)

        save_btn = tk.Button(btn_frame, text="Save Image", command=self.save_image, width=15)
        save_btn.grid(row=0, column=2, padx=5)

    def load_image(self):
        # Open a file dialog to select an image
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not self.image_path:
            return

        # Load the image using OpenCV
        self.original_image = cv.imread(self.image_path)
        self.original_image = cv.cvtColor(self.original_image, cv.COLOR_BGR2RGB)

        # Display the image on the canvas
        self.display_image(self.original_image)

    def display_image(self, image):
        # Resize the image to fit the canvas
        resized_image = self.resize_image(image, 800, 600)
        
        # Convert the image to PIL format
        im = Image.fromarray(resized_image)
        imgtk = ImageTk.PhotoImage(image=im)

        self.canvas.image = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def resize_image(self, image, width, height):
        h, w, _ = image.shape
        scale = min(width / w, height / h)
        resized = cv.resize(image, (int(w * scale), int(h * scale)))
        return resized

    def color_correct(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        # Convert the image to LAB color space
        lab_image = cv.cvtColor(self.original_image, cv.COLOR_RGB2LAB)
        l, a, b = cv.split(lab_image)

        # Apply CLAHE with a milder clip limit
        clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_corrected = clahe.apply(l)

        # Apply gamma correction (optional)
        gamma = 1.2
        l_corrected = np.array(((l_corrected / 255.0) ** (1 / gamma)) * 255, dtype=np.uint8)

        # Merge the corrected L-channel back with A and B channels
        corrected_lab = cv.merge((l_corrected, a, b))

        # Convert back to RGB color space
        corrected_rgb = cv.cvtColor(corrected_lab, cv.COLOR_LAB2RGB)

        # Optional: Blend the corrected image with the original for a softer look
        alpha = 0.7
        self.corrected_image = cv.addWeighted(corrected_rgb, alpha, self.original_image, 1 - alpha, 0)

        # Display the corrected image
        self.display_image(self.corrected_image)

    def save_image(self):
        if self.corrected_image is None:
            messagebox.showerror("Error", "No corrected image to save!")
            return

        # Open a file dialog to save the image
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")])
        if not save_path:
            return

        # Convert corrected image to BGR format for saving with OpenCV
        corrected_bgr = cv.cvtColor(self.corrected_image, cv.COLOR_RGB2BGR)
        cv.imwrite(save_path, corrected_bgr)

        messagebox.showinfo("Saved", f"Image saved successfully at {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorCorrectorApp(root)
    root.mainloop()
