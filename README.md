# Growing Neural Cellular Automata
Morphogenesis is the process through which an organism develops its shape, whereby cells communicate with one another to determine organ and body structure.
Modelling biological morphogenesis in computation can allow us to gain a better understanding of how the shape and form of an organism develops. Such models both enhance our understanding of biology and translate these discoveries into improved robotics and computational technology. Models that can define global coordination out of local level interactions can serve as a valuable tool for biologists, helping to deepen their understanding of morphogenesis and its underlying mechanisms. Additionally, by using these models to study the development of organisms, it may be possible to create more advanced computational technology that can replicate the complexity and functionality of natural systems.

## Algorithm Details 
### Voxel Generation
A voxel dataset is essentially a three-dimensional grid of small, cubic elements called voxels, each of which can have a color value associated with it.We provide a CSV file containing values of dimensions and RGB colors as input. The dimensions of the voxel model are extracted from the file name, and the maximum dimension is also calculated.The CSV file contains a list of coordinates along with their corresponding RGB color values. Each row in the CSV file corresponds to a single voxel in the 3D voxel dataset. The coordinates and color values are typically separated by commas, which allows them to be easily parsed and loaded into a program. The stepwise voxel generation process can be demonstrated as:

### Reading and loading the CSV file:
The CSV file contains the list of coordinates along with their corresponding RGB color values

### Returning a single layer and stacking up the values: 
In order to visualise a 3D voxel dataset, we need to extract each layer of the dataset and stack them up to form the entire 3D voxel.

### Model Training:
![model](https://user-images.githubusercontent.com/64689273/234799844-f2e0962c-f76c-4875-94e5-5df21d3288d0.png)

### Target Object:

![target](https://user-images.githubusercontent.com/64689273/234800036-1a72a9fa-5963-463b-a195-74e6b87f787a.png)

### Input: seed value

![seed](https://user-images.githubusercontent.com/64689273/234800180-b3e68cf1-5270-4d90-b8fb-604504a0c634.png)

### Output: 
The output after training: 



https://user-images.githubusercontent.com/64689273/234800590-c102695c-37cc-40d6-b56d-7927c634d255.mp4

### Gradio is used for the user interface. 
![ui](https://user-images.githubusercontent.com/64689273/234802750-b7bfae08-08a7-4a80-a1d3-a95850e8c005.png)


## To run the model:
1. Create a virtual environment
2. Install requirements.txt
3. Create a folder named img
4. Run app.py




