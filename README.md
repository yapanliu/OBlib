## Occupant Behavior Library for commercial buildings using deep learning methods

This repository contains different OB model such as window opening, Occupancy, Lightning, Shading, Thermostat, Appliances and corresponding pre-release codes 

## Citing:
tba

## Dependencies

* python 3 (Version..)
* tensorflow (tested on .. )



## How to use model?

tba

## Citing:
Markovic, R., Grintal, E., Wölki, D., Frisch, J., & van Treeck, C. (2018). Window opening model using deep learning methods. Building and Environment, 145, 319-329.

## Prerequisite

In order to run the code, your setup has to meet the following minimum requirements (tested versions in parentheses. Other versions might work, too):

* python 3.6
* tensorflow 1.9.0

## Create conda environment

* conda create -n OBLib python=3.6
* activate OBLib

## How to use model deep learning model?

* make sure that required python and tensorflow versions are installed
* clone the repository using following command:
0. `git clone https://git.rwth-aachen.de/romana.markovic/window_opening`
* define the path where the repository is cloned in main.py, line 30
* save the required input data in ~/input_data; alternatively defined path to the input data
* define path to your adaptation sets in main.py lines 32-35
* uncomment line 141 OR 142, depending of execution modus
* open terminal and change path to ~/main
0. `cd path-to-repository/window_opening/main`
* run evaluation from terminal using following command
0. `python main.py`
* the generated window states will be saved in a .txt file in ~/output


## License

GNU General Public License (http://www.gnu.org/licenses/gpl.html)

Copyright (c) 2018 Romana Markovic
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



