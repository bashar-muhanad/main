import cv2
import os
import base64
import numpy as np
from pyzbar import pyzbar
import mysql.connector as sql
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from jnius import autoclass
#tesseract_executable_path = os.path.join(os.environ['HOME'], 'tesseract-ocr', 'tesseract')
#pytesseract.pytesseract.tesseract_cmd = tesseract_executable_path
Activity = autoclass("org.kivy.android.PythonActivity").mActivity
def calculate_and_print_invoice(qr_code1):
    qr_code = qr_code1[0]  # استخراج القيمة من القائمة
    print(qr_code)
    
    try:
        con = sql.connect(
            host='localhost',
            user='root',
            password='',
            database='kivo'
        )
        cur = con.cursor()

        # البحث عن المستخدم باستخدام تسلسل QR
        query = 'SELECT id, username, address, phone, oldnumber, dateoldnumber, number, datenumber, newnumber, datenewnumber FROM users2 WHERE id = %s'
        cur.execute(query, (qr_code,))
        result = cur.fetchone()
        
        if result:
            user_id, username, address, phone, oldnumber, dateoldnumber, number, datenumber, newnumber, datenewnumber = result

            # تحويل القيم النصية إلى أرقام صحيحة
            oldnumber = int(oldnumber)
            number = int(number)

            # حساب الفاتورة
            difference = number + oldnumber
            invoice_amount = difference * 160

            # تحديث الحقول المناسبة
            update_query = '''
            UPDATE users2
            SET newnumber = %s
            WHERE id = %s
            '''
            cur.execute(update_query, (invoice_amount, user_id))
            con.commit()
            print("Invoice calculated and user updated successfully.")
            
            # طباعة الفاتورة بشكل منظم
            print("\n--- Invoice ---")
            print(f"ID: {user_id}")
            print(f"Username: {username}")
            print(f"Address: {address}")
            print(f"Phone: {phone}")
            print(f"Old Number: {oldnumber}")
            print(f"Date Old Number: {dateoldnumber}")
            print(f"Number: {number}")
            print(f"Date Number: {datenumber}")
            print(f"New Number: {invoice_amount}")
            print(f"Date New Number: {datenewnumber}")
            print("----------------\n")
        else:
            print("User not found.")
    
    except sql.Error as e:
        print(f"Error: {e}")
    
    finally:
        if con.is_connected():
            cur.close()
            con.close()
            print("Connection closed.")

def update_user_with_qr(qr_code1, new_number):
    qr_code = qr_code1[0]  # استخراج القيمة من القائمة
    print(qr_code)
    print('qss', qr_code, 'newqs', new_number)
    try:
        con = sql.connect(
            host='localhost',
            user='root',
            password='',
            database='kivo'
        )
        cur = con.cursor()

        # البحث عن المستخدم باستخدام تسلسل QR
        query = 'SELECT id, number FROM users2 WHERE id = %s'
        cur.execute(query, (qr_code,))
        result = cur.fetchone()
        print('qs', qr_code)
        if result:
            user_id, current_number = result

            # تحديث الحقول المناسبة
            update_query = '''
            UPDATE users2
            SET oldnumber = %s, number = %s
            WHERE id = %s
            '''
            cur.execute(update_query, (current_number, new_number, user_id))
            con.commit()
    except sql.Error as e:
        print(f"Error: {e}")
    finally:
        if con.is_connected():
            cur.close()
            con.close()

def convlutionmwthod(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # اكتشاف الحواف
    edges = cv2.Canny(gray, 100, 300, apertureSize=3)

    # اكتشاف الخطوط باستخدام Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # حساب زاوية الميل
    angle = 0
    if lines is not None:
        for rho, theta in lines[0]:
            angle = (theta * 180 / np.pi) - 90

    # تدوير الصورة
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    #cv2.imshow('Processed rotated', rotated)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return rotated

def edgeexcelant(image):
    # تحويل الصورة إلى تدرجات الرمادي لاكتشاف الحواف
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # تطبيق فلتر Gaussian Blur لتقليل الضوضاء
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # إنشاء فلتر الشحذ
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # تطبيق الفلتر على الصورة
    sharpened = cv2.filter2D(image, -1, kernel)
    # استخدام Canny Edge Detection لاكتشاف الحواف
    edges = cv2.Canny(sharpened, 100, 300)

    # تحويل الحواف إلى صورة ملونة
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # دمج الحواف مع الصورة الأصلية لجعل الحدود أكثر حدة
    sharp_image = cv2.addWeighted(image, 0.5, edges_colored, 0.5, 0)

    return sharp_image

def enhance_red_lines(image):
    # تحويل الصورة إلى تدرجات الرمادي
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    red_only = cv2.bitwise_and(image, image, mask=mask)

    hsv_red = cv2.cvtColor(red_only, cv2.COLOR_BGR2HSV)
    hsv_red[:, :, 1] = cv2.add(hsv_red[:, :, 1], 50)  # زيادة التشبع
    hsv_red[:, :, 2] = cv2.add(hsv_red[:, :, 2], 50)  # زيادة السطوع

    enhanced_red = cv2.cvtColor(hsv_red, cv2.COLOR_HSV2BGR)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    enhanced_red = cv2.morphologyEx(enhanced_red, cv2.MORPH_CLOSE, kernel)

    return enhanced_red

def exctract_number(imageexc):
    gray = cv2.cvtColor(imageexc, cv2.COLOR_RGB2GRAY)
    bilateral_filtered_image1 = cv2.bilateralFilter(gray, 400, 30, 30)
    
    # استخدام Canny Edge Detection لاكتشاف الحواف
    edges = cv2.Canny(bilateral_filtered_image1, 50, 150)
    
    # تحويل الحواف إلى صورة ملونة
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # دمج الحواف مع الصورة الأصلية لجعل الحدود أكثر حدة
    sharp_image1 = cv2.addWeighted(imageexc, 0.8, edges_colored, 0.3, 0)
    gray_sharp_image = cv2.cvtColor(sharp_image1, cv2.COLOR_BGR2GRAY)

    _, thresh1 = cv2.threshold(gray_sharp_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    eroded1 = cv2.erode(thresh1, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT)

    dilated1 = cv2.dilate(eroded1, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated1, cv2.MORPH_CLOSE, kernel,iterations=3)
    retval, buffer = cv2.imencode('.jpg', closed)
    #cv2.imshow('exctract number', closed)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    text = Activity.tess_text(base64.b64encode(buffer))
    App.get_running_app().label.text = text
    Activity.toast(text)
    numbers = ''.join(filter(str.isdigit, text))

    print('text',numbers[:5])
    if len(numbers) >= 5:
                return numbers[:5]
    else:
        return print("I'm sorry")

def find_optimal_sharpening(image):
    def measure_readability(image):
        # تحويل الصورة إلى تدرجات الرمادي
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # استخدام Tesseract لقراءة النصوص
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        retval, buffer = cv2.imencode('.jpg', image)
        Activity.toast("Processing")
        text = Activity.tess_text(base64.b64encode(buffer))
        App.get_running_app().label.text = text
        Activity.toast(text) 
        #text = pytesseract.image_to_string(gray, config=custom_config)
        print("textsharpen", text)
        # حساب عدد الأرقام في النص
        digits = [char for char in text if char.isdigit()]
        return len(digits)

    # تحسين التباين باستخدام CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    best_readability = 0
    best_amount = 0
    for amount in np.arange(0.1, 2.0, 0.1):
        # إنشاء فلتر الشحذ
        kernel = np.array([[0, -1, 0],
                           [-1, 5 + amount, -1],
                           [0, -1, 0]])
        # تطبيق الفلتر على الصورة
        sharpened = cv2.filter2D(enhanced_image, -1, kernel)
        # قياس قابلية القراءة
        readability = measure_readability(sharpened)
        if readability >= 5:  # تحقق من قراءة خمسة أرقام على الأقل
            best_readability = readability
            best_amount = amount
            break

    # تطبيق الشحذ باستخدام القيمة المثلى
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + best_amount, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(enhanced_image, -1, kernel)
    #cv2.imshow('sharpened', sharpened_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return sharpened_image, best_amount

def read_qr_code(image):
    # قراءة الصورة
    qr_codes = pyzbar.decode(image)

    # استخراج البيانات من رموز QR
    qr_data = []
    for qr_code in qr_codes:
        qr_data.append(qr_code.data.decode('utf-8'))

    return qr_data

def process_camera_frame(frame):
    # تغيير حجم الصورة إلى 680x480
    image = cv2.resize(frame, (680, 480))

    # تقسيم الصورة إلى ثلاثة أجزاء واستخراج الجزء العلوي
    height, width = image.shape[:2]
    upper_part = image[:height//3, :]
    lower_part = image[height//3:, :]
    qr = read_qr_code(lower_part)
    print("qr", qr)
    sharpened_image, best_amount = find_optimal_sharpening(upper_part)
    image1 = convlutionmwthod(sharpened_image)
    image2 = edgeexcelant(image1)
    image3 = enhance_red_lines(image2)

    # تحويل الجزء العلوي إلى نظام الألوان HSV
    hsv = cv2.cvtColor(upper_part, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) >= 2:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        x1, y1, w1, h1 = cv2.boundingRect(contours[0])
        x2, y2, w2, h2 = cv2.boundingRect(contours[1])
        if 5 <= h1 <= 45 and 5 <= h2 <= 45:
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
            cropped_image1 = upper_part[y:y+h, x:x+w]
            exctract_numerical = exctract_number(cropped_image1)
            update_user_with_qr(qr, exctract_numerical)
            calculate_and_print_invoice(qr)
    else:
        gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        bilateral_filtered_image = cv2.bilateralFilter(gray, 300, 30, 30)
        _, thresh = cv2.threshold(bilateral_filtered_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        eroded = cv2.erode(thresh, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        edges = cv2.Canny(dilated, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            valid_rectangle = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if 200 <= w <= 450 and 35 <= h <= 60:
                    valid_rectangle.append((x, y, w, h))
            if valid_rectangle:
                for rect in valid_rectangle:
                    x, y, w, h = rect
                    cropped_image = upper_part[y:y+h, x:x+w]
                    exc_num = exctract_number(cropped_image)
                    update_user_with_qr(qr, exc_num)

class CameraApp(App):
    def on_resume(self):
        if(Activity.tmp != ""):
            self.capture()
    def build(self):
        self.label = Label(text="Restart app to run again")
        #Activity.toast(Activity.tmp)
        Activity.camera()
        """layout = BoxLayout(orientation='vertical')
        self.camera = Camera(play=True, resolution=(640, 480))
        layout.add_widget(self.camera)
        btn = Button(text="Capture", size_hint=(1, 0.2))
        btn.bind(on_press=self.capture)
        layout.add_widget(btn)"""
        return self.label

    def capture(self):
        image_data = base64.b64decode(Activity.tmp)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        '''frame = self.camera.texture.pixels
        frame = np.frombuffer(frame, np.uint8).reshape(self.camera.texture.height, self.camera.texture.width, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)'''
        process_camera_frame(frame)

if __name__ == '__main__':
    CameraApp().run()
