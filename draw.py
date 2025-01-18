import numpy as np
import img_predict as impd
import tkinter as tk
from PIL import Image, ImageDraw

"""
drawing app file
this is where the drawing gui is done

"""

class DrawingApp:
    def __init__(self, root, model_name):
        self.root = root
        root.title("Digit Prediction")
        
        self.model_name = model_name

        self.canvas = tk.Canvas(root, width=280, height=280, bg='black')  
        self.canvas.pack()

        self.last_x, self.last_y = None, None
        self.pen_color = "#FFFFFF"  

        # blank image where drawing is stored for processing
        self.image = Image.new("L", (280, 280), "black") 
        self.draw = ImageDraw.Draw(self.image) 

        # mouse inputs
        self.canvas.bind("<B1-Motion>", self.paint) 
        self.canvas.bind("<ButtonRelease-1>", self.predict_drawing)

        # labels
        self.label = tk.Label(root, text="Draw a digit", font=("Arial", 14))
        self.label.pack()

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=20, fill=self.pen_color, capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, x, y], fill="white", width=20)
        self.last_x, self.last_y = x, y

    def save_image(self, event):
        self.last_x, self.last_y = None, None
        # Save the image to a file
        self.image.save("drawing.png")
        print("Drawing saved as 'drawing.png'")
        self.clear_canvas()

    def predict_drawing(self, event):
        self.last_x, self.last_y = None, None

        # resize and normalize drawn image
        img_resized = self.image.resize((28, 28))
        img_array = np.array(img_resized).astype("float32") / 255 
        img_array = np.reshape(img_array, (1, 28 * 28)) 
        img_flat = img_array.reshape(784,)

        # predict with model, then find confidence of prediction
        output = impd.predict(img_flat,self.model_name) 
        predicted_digit = np.argmax(output) 
        confidence = (output[predicted_digit].item() / np.sum(output).item()) * 100
        confidence = f"{confidence:.1f}" 

        # display prediction and confidence
        self.label.config(text=f"Predicted Digit: {predicted_digit}\nConfidence: {confidence}%")
        

    def clear_canvas(self):
        """Reset the canvas and the image"""
        self.canvas.delete("all")  
        self.image = Image.new("L", (280, 280), color="black") 
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Draw a digit")  

    

def predict(model_name):
    root = tk.Tk()    
    app = DrawingApp(root, model_name)
    
    clear_button = tk.Button(root, text="Clear", command=app.clear_canvas)
    clear_button.pack()
    
    root.mainloop()

# used for troubleshooting purposes
def save_drawn_image(img):
        """Save the drawn image to a file."""
        img_pil = Image.fromarray(img)
        img_pil.save("drawn_image.png") 
        
        print("Image saved as drawn_image.png")