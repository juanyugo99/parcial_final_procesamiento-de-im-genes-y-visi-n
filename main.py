from banderas import *
import os

if __name__ == "__main__":

    path = "C:/Users/Juan Pablo/Im_Procesamiento/flags/"
    which = input("que bandera quiere escoger?(1-5): ")
    image_path = "flag{}.png".format(which)
    path = os.path.join(path,image_path)
    flag = cv2.imread(path)
    bandera = banderas(flag)
    bandera.colores()
    print(bandera.porcentaje())