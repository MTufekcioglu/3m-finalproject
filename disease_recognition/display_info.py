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
plant_info = {'Apple' : [(0, 35), (450, 900)],
              'Blueberry': [(5, 30), (450, 900)], 
              'Cherry' : [(5, 35), (450, 900)],
              'Corn' : [(20, 41), (300, 900)], 
              'Grape' : [(5, 35), (550, 900)], 
              'Orange' : [(20, 38), (450, 900)],
              'Peach' : [(5, 35), (450, 900)], 
              'Bell Pepper' : [(20, 38), (650, 900)], 
              'Potato' : [(20, 35), (550, 900)], 
              'Raspberry' : [(20, 35), (450, 900)], 
              'Soybean' : [(20, 38), (650, 900)],
              'Squash' : [(20, 38), (650, 900)], 
              'Strawberry' : [(5, 32), (650, 900)], 
              'Tomato' : [(20, 35), (650, 900)],
              'Basil' : [(20, 38), (650, 900)]}

# Initialize timer
start_time = time.time()

# Set a flag so we identify the plant on the first loop iteration
first = 1

while True:
        # Read moisture
        touch = ss.moisture_read()

        # Read temp
        temp =ss.get_temp()
        temp = round(temp, 2)

        ### Performing the classification every 6 hours ###

        curr_time = time.time()
        if (curr_time - start_time > 21600 or first == 1):
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

                # Disable the first loop iteration flag
                first = 0
       
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


