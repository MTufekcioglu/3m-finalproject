import pyfiglet 
import time
from predict_disease import *

from board import SCL, SDA
import busio

from adafruit_seesaw.seesaw import Seesaw

i2c_bus = busio.I2C(SCL, SDA)

ss = Seesaw(i2c_bus, addr=0x36)

# Loading in ResNet9 model
model_path = './model.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Dictionary of species-specific care need thresholds 
# Each entry formatted as [(temp_low, temp_high), (moist_low, moist_high)]
# Thresholds estimated based on information from https://www.picturethisai.com/
plant_info = {'Apple' : [(0, 35), (450, 800)],
              'Blueberry': [(5, 30), (450, 800)], 
              'Cherry' : [(5, 35), (450, 800)],
              'Corn' : [(20, 41), (300, 800)], 
              'Grape' : [(5, 35), (550, 800)], 
              'Orange' : [(20, 38), (450, 800)],
              'Peach' : [(5, 35), (450, 800)], 
              'Bell Pepper' : [(20, 38), (650, 800)], 
              'Potato' : [(20, 35), (550, 800)], 
              'Raspberry' : [(20, 35), (450, 800)], 
              'Soybean' : [(20, 38), (650, 800)],
              'Squash' : [(20, 38), (650, 800)], 
              'Strawberry' : [(5, 32), (650, 800)], 
              'Tomato' : [(20, 35), (650, 800)],
              'Basil' : [(20, 38), (650, 800)]}

# Initialize timer
start_time = time.time()

while True:
        # Read moisture
        touch = ss.moisture_read()

        # Read temp
        temp =ss.get_temp()
        temp = round(temp, 2)

        ### Performing the classification every 6 hours ###

        curr_time = time.time()
        if (curr_time - start_time > 21600):
                 # Getting the current images folder
                images = ImageFolder('./images/', transform=transforms.ToTensor())

                # Running classification to get species and health
                species, health = predict_image_pretty(images[0], model)

                # Setting the species-specific thresholds
                plant_needs = plant_info[species]
                thresh_low_temp, thresh_high_temp = plant_needs[0]
                thresh_low_moist, thresh_high_moist = plant_needs[1]

                # Reset timer
                start_time = time.time()
       
        ### End of classification segment ###


        # Create figlet strings
        temp_too_cold = pyfiglet.figlet_format("Temp: " + str(temp) + "   Too Cold!", width = 1000)
        temp_too_warm = pyfiglet.figlet_format("Temp: " + str(temp) + "   Too Warm!", width = 1000)
        temp_fine = pyfiglet.figlet_format("Temp: " + str(temp))
        need_water = pyfiglet.figlet_format("Moisture:  " + str(touch) + "   Water Me!", width = 1000) 
        stop_water = pyfiglet.figlet_format("Moisture:  " + str(touch) + "   Too Much Water!", width = 1000) 
        water_fine = pyfiglet.figlet_format("Moisture: " + str(touch))
        species_str = pyfiglet.figlet_format("Species:  " + species, width = 1000)
        healthy  = pyfiglet.figlet_format("Health:  " + health, width = 1000)
        not_healthy = pyfiglet.figlet_format("Health:  " + "Possible  " + health, width = 1000)
        
        #Clear Screen
        print("\x1b[2J")
        

        ### Conditionally format and print figlet strings ###

        # Species
        print(species_str)

         # Health
        if health == 'Healthy':
                print("\x1b[38;5;113m" + healthy + "\x1b[0m")
        else:
                print("\x1b[38;5;167m" + not_healthy + "\x1b[0m")

        # Temperature
        if temp < thresh_low_temp:
                print("\x1b[38;5;81m" + temp_too_cold + "\x1b[0m")
        elif temp > thresh_high_temp:
                print("\x1b[38;5;167m" + temp_too_warm + "\x1b[0m")
        else:
                print("\x1b[38;5;113m" + temp_fine + "\x1b[0m")

        # Moisture
        if touch < thresh_low_moist:
                print("\x1b[38;5;81m" + need_water + "\x1b[0m")
        elif touch > thresh_high_moist:
                print("\x1b[38;5;167m" + stop_water + "\x1b[0m")
        else:
                print("\x1b[38;5;113m" + water_fine + "\x1b[0m")


        # Add a delay between reading measurements
        time.sleep(1)


