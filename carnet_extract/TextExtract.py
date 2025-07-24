from PIL import Image
from pdf2image import convert_from_path
import pytesseract

'''
pages = convert_from_path("factura.pdf")
texto = pytesseract.image_to_string(pages[0], lang='spa')
print(texto)
'''

def extractTextFromPDF(pdfPath):
    pages = convert_from_path(pdfPath)
    print(pages)
    texto = pytesseract.image_to_string(pages[0], lang='spa')
    return texto

def extractTextFromImage(imagePath):
    image = Image.open(imagePath)
    texto = pytesseract.image_to_string(image, lang='spa')
    return texto

'''
print('IMAGE**********************************************')
print(extractTextFromImage(imagePath='factura.jpg'))

print('PDF**********************************************')
print(extractTextFromPDF(pdfPath='factura.pdf'))
'''
#print(extractTextFromImage('carnet_recortado.jpg'))