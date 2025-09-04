from io import BytesIO
from PIL import Image, ImageDraw
import pandas as pd
from src.document_ingestion.data_ingestion import ImageHandler,TableHandler

def create_test_image():
    img = Image.new("RGB", (200, 100), color="blue")
    d = ImageDraw.Draw(img)
    d.text((10, 40), "Test", fill="white")
    buf = BytesIO()
    buf.name = "test.png"  # simulate uploaded_file with name
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf
def create_test_csv():
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "City": ["NY", "LA", "Chicago"]
    })
    buf = BytesIO()
    buf.name = "test.csv"   # simulate uploaded_file
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

if __name__ == "__main__":
    image_handler = ImageHandler()
    uploaded_image = create_test_image()
    path = image_handler.save_file(uploaded_image)
    print("Saved at:", path)

    metadata = image_handler.read_file(path)
    print("Metadata:", metadata)

    table_handler = TableHandler()
    uploaded_file = create_test_csv()
    saved_path = table_handler.save_file(uploaded_file)
    print("Saved at:", saved_path)
    
    # Read the table (metadata + preview)
    table_info = table_handler.read_file(saved_path)
    print("Table Info:", table_info)
