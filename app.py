import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

model=YOLO(r'C:\Users\kemal\Desktop\kırık_tespiti\best.pt')

st.title("KARAR DESTEK SİSTEMİ")

with st.sidebar:
    st.header("Projemiz Hakkımızda")
    st.markdown(
        """Cumhurbaşkanlığı Dijital Dönüşüm Ofisi'nin Genç Yetenek Akademisi'nin yapay zeka eğitim programı kapsamında geliştirdiğimiz bu proje, doktorlara X-ray görüntülerinden el ve kol kemiklerinde kırık tespiti konusunda karar destek sağlayacak bir sistem sunmaktadır.

Modelimiz, derin öğrenme tekniklerini kullanarak yüksek doğruluk oranıyla kırıkların otomatik olarak tespit edilmesini sağlar. Bu sayede, doktorlar daha hızlı ve güvenilir bir şekilde teşhis koyabilir ve tedavi süreçlerini daha etkili bir şekilde yönetebilirler.



**Projenin Önemi:** Bu sistem, hem acil hem de rutin X-ray taramalarında doktorlara ek bir araç sunarak teşhis süreçlerini destekler. Ayrıca, erken teşhis ve doğru tedavi ile hastaların iyileşme süreçlerini hızlandırabilir.""")

st.header("Kırık Tespit Etmek İstediğiniz Görüntüyü Yükleyiniz ")
uploaded_files = st.file_uploader(".", type=["jpg", "png", "jpeg"])

if uploaded_files is not None:
    image = Image.open(uploaded_files)
    image_resized = image.resize((640, 640))  # Model 640x640 çözünürlükte eğitildi

    # İki sütun oluşturuyoruz
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_resized, caption='Yüklenen Görüntü', use_column_width=True)

    img_array = np.array(image)
    results = model(img_array)

    boxes = results[0].boxes  # Tahmin edilen kutuları al
    if boxes.shape[0] > 0:
        st.success("Kırık tespit edildi!")
        for box in boxes:
            b = box.xyxy[0].cpu().numpy()  # Kutu koordinatlarını al
            label = f'{model.names[int(box.cls)]} {float(box.conf):.2f}'  # Sınıf etiketi ve güven puanı float olarak alınıyor
            cv2.rectangle(img_array, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
            cv2.putText(img_array, "Kirik", (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        with col2:
            st.image(img_array, caption='Kırık Tespit Sonuçları', use_column_width=True)
    else:
        st.warning("Kırık bulunamadı.")

