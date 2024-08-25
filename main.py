from kivy.app import App
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.uix.camera import Camera
from kivy.uix.filechooser import FileChooserIconView
import mysql.connector as sql
import cv2
import numpy as np
import pytesseract

Window.clearcolor = (0, 60/255.0, 1, 0)  # r g b  
Window.size = (370, 600)

class app1(App):
    def build(self):
        self.title = 'reading meter'
        layout = GridLayout(cols=1)
        self.imo = Image(source='2.jpg')
        self.L1 = Label(text='meTer')
        self.L2 = Label(text='Add new employee')
        self.id = TextInput(hint_text='ID')
        self.username = TextInput(hint_text='Add username')
        self.work = TextInput(hint_text='Work')
        self.phone = TextInput(hint_text='Phone')
        self.numbernew = TextInput(hint_text='Number new')
        self.note = TextInput(hint_text='Note')
        self.camera = Camera(play=True)
        submit = Button(text='Add employee', on_press=self.sub)
        submit1 = Button(text='Update number by name', on_press=self.sub1)
        submit3print = Button(text='Print', on_press=self.sub2)
        capture_button = Button(text='Capture and Process', on_press=self.capture_and_process_image)
        choose_button = Button(text='Choose Image from Gallery', on_press=self.choose_image)
        
        layout.add_widget(self.imo)
        layout.add_widget(self.L1)
        layout.add_widget(self.L2)
        layout.add_widget(self.id)
        layout.add_widget(self.username)
        layout.add_widget(self.work)
        layout.add_widget(self.phone)
        layout.add_widget(self.numbernew)
        layout.add_widget(submit)
        layout.add_widget(submit1)
        layout.add_widget(submit3print)
        layout.add_widget(self.camera)
        layout.add_widget(capture_button)
        layout.add_widget(choose_button)
        
        return layout
    
    def sub(self, ob):
        id1 = self.id.text
        usename = self.username.text
        wor = self.work.text
        phon = self.phone.text
        numbero = self.numbernew.text
        number = self.numbernew.text
        note = self.note.text
        con = sql.connect(host='localhost', user='root', password='', database='kivo')
        cur = con.cursor()
        query = 'INSERT INTO users1(id, username, work, phone, numberold, numbernew) VALUES(%s, %s, %s, %s, %s, %s)'
        val = (id1, usename, wor, phon, numbero, number)
        cur.execute(query, val)
        con.commit()
        con.close()
    
    def sub1(self, upd):
        nam = self.username.text
        con = sql.connect(host='localhost', user='root', password='', database='kivo')
        cur = con.cursor()
        query = 'SELECT * FROM users1 WHERE username = %s'
        val = (nam,)
        cur.execute(query, val)
        rows = cur.fetchall()
        if rows:
            query = 'UPDATE users1 SET numbernew = %s WHERE username = %s'
            val = (self.numbernew.text, nam)
            cur.execute(query, val)
            con.commit()
            con.close()
    
    def sub2(self, upd):
        nam = self.username.text
        con = sql.connect(host='localhost', user='root', password='', database='kivo')
        cur = con.cursor()
        query = 'SELECT * FROM users1 WHERE username = %s'
        val = (nam,)
        cur.execute(query, val)
        rows = cur.fetchall()
        if rows:
            query = 'UPDATE users1 SET numbernew = %s WHERE username = %s'
            val = (self.numbernew.text, nam)
            cur.execute(query, val)
            con.commit()
            self.print_invoice(rows)
        con.close()

    def print_invoice(self, user_data):
        numold = int(user_data[0][4])
        numnew = int(user_data[0][5])
        defr = numnew - numold
        print("فاتورة")
        print("ID Customer:", user_data[0][0])
        print("Name Customer:", user_data[0][1])
        print("Number Old:", numold)
        print("Number New:", numnew)
        print("Difference:", defr)
    
    def capture_and_process_image(self, instance):
        self.camera.export_to_png("captured_image.png")
        print("Image captured and saved as captured_image.png")
        self.process_image("captured_image.png")

    def choose_image(self, instance):
        filechooser = FileChooserIconView()
        filechooser.bind(on_selection=self.process_selected_image)
        self.root.add_widget(filechooser)

    def process_selected_image(self, filechooser, selection):
        if selection:
            image_path = selection[0]
            print(f"Image selected: {image_path}")
            self.process_image(image_path)
            self.root.remove_widget(filechooser)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        height = image.shape[0]
        third_height = height // 3
        top_part = image[:third_height, :]
        bottom_part = image[2*third_height:, :]
        
        # استخراج الرقم من الجزء العلوي
        top_text = pytesseract.image_to_string(top_part, config='--psm 6')
        top_number = ''.join(filter(str.isdigit, top_text))
        
        # استخراج الرقم من الجزء السفلي
        bottom_text = pytesseract.image_to_string(bottom_part, config='--psm 6')
        bottom_number = ''.join(filter(str.isdigit, bottom_text))
        
        # تحديث قاعدة البيانات
        con = sql.connect(host='localhost', user='root', password='', database='kivo')
        cur = con.cursor()
        
        # تحديث numberold و numbernew
        query = 'UPDATE users1 SET numberold = numbernew, numbernew = %s WHERE id = %s'
        val = (top_number, bottom_number)
        cur.execute(query, val)
        con.commit()
        con.close()
        
        print("Database updated with new numbers.")
        
        # تطبيق فلتر العتبة (threshold) وفلتر Canny على الجزء العلوي
        _, thresholded = cv2.threshold(top_part, 128, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresholded, 100, 200)
        
        # حفظ النتائج
        cv2.imwrite('top_part.png', top_part)
        cv2.imwrite('thresholded.png', thresholded)
        cv2.imwrite('edges.png', edges)
        print("Image processing completed and results saved.")

if __name__ == '__main__':
    app1().run()
