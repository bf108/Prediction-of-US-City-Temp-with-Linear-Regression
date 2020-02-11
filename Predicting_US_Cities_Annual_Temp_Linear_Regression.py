import numpy as np
import pylab
import matplotlib.pyplot as plt
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

INTERVAL_1 = list(range(1961, 1985))
INTERVAL_2 = list(range(1985, 2016))

class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a numpy 1-d array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):#change back to 32 for full month
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]
    
    def get_monthly_mean(self, city, month, year):
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        
        mean_temp = []
        
        for day in range(1,32):#change back to 32 for full month
            if day in self.rawdata[city][year][month]:
                mean_temp.append(self.rawdata[city][year][month][day])
                
        mean = sum(mean_temp)/float(len(mean_temp))
        return mean


# Problem 1
def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).
    Args:
        x: a list with length N, representing the x-coords of N sample points
        y: a list with length N, representing the y-coords of N sample points
        degs: a list of degrees of the fitting polynomial
    Returns:
        a list of numpy arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    # TODO
    models = []
    xval = np.array(x)
    yval = np.array(y)
    for d in degs:
        models.append(pylab.polyfit(xval, yval, d))
    
    return models

# Problem 2
def r_squared(observed, predicted):
    """
    Calculate the R-squared error term.
    Args:
        y: list with length N, representing the y-coords of N sample points
        estimated: a list of values estimated by the regression model
    Returns:
        a float for the R-squared error term
    """
    # TODO
    error = ((predicted - observed)**2).sum()
    meanError = error/len(observed)
    return 1 - (meanError/np.var(observed))

# Problem 3
def evaluate_models_on_training(x, y, models,dataSet):
    """
    For each regression model, compute the R-square for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points
    Args:
        x: a list of length N, representing the x-coords of N sample points
        y: a list of length N, representing the y-coords of N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        dataSet: String outlining data set in analysis e.g Boston, Chicago or USA
    Returns:
        None
    """
    # TODO
    styles = ['r-','g-','y-']
    pylab.plot(x,y,'b-',label = 'Mean Temperature Data')
    # rSquares = {}
    i = 0
    for m in models:
        # rSquares[m] = []
        estYVals = pylab.polyval(m, np.array(x))
        # rSquares[m].append(estYVals)
        error = round(r_squared(np.array(y), estYVals),5)
        pylab.plot(x,estYVals,styles[i],label = (str(str(i+1) + "Order Polynomial" + "\n" + " R-Squared Error: " + str(error))))
        i +=1
    
    pylab.title('Average Annual {} Temperatures'.format(dataSet))
    pylab.xlabel('Years')
    pylab.ylabel('Average Temperature Deg C')
    pylab.legend(loc = 'best')

### Begining of program


'''

Code to plot avg annual temp for all US cities 

The regression models are show on the plots also with respective errors

The linear regression appears to be much more accurate than the ploynomials versions

'''
def usAverageTemp():
    raw_data = Climate('data.csv')
    avg_temp1 = []
    for year in INTERVAL_1:
        avg = 0
        for city in CITIES:
            avg += sum(raw_data.get_yearly_temp(city,year))/float(len(raw_data.get_yearly_temp(city,year)))
        avg_temp1.append(avg/float(len(CITIES)))
        
    avg_temp2 = []
    for year in INTERVAL_2:
        avg = 0
        for city in CITIES:
            avg += sum(raw_data.get_yearly_temp(city,year))/float(len(raw_data.get_yearly_temp(city,year)))
        avg_temp2.append(avg/float(len(CITIES)))
        
    models = generate_models(INTERVAL_1,avg_temp1,[1,2])
    evaluate_models_on_training(INTERVAL_2,avg_temp2,models,'USA')
    
def cityAverageTemp(city):
    city = city.upper()
    while city not in CITIES:
        [print(c) for c in CITIES]
        city = input('City not available, select a city from above list: ').upper()
    raw_data = Climate('data.csv')
    avg_temp1 = []
    for year in INTERVAL_1:
        avg_temp1.append(sum(raw_data.get_yearly_temp(city,year))/float(len(raw_data.get_yearly_temp(city,year))))
        
    avg_temp2 = []
    for year in INTERVAL_2:
        avg_temp2.append(sum(raw_data.get_yearly_temp(city,year))/float(len(raw_data.get_yearly_temp(city,year))))
    
    models = generate_models(INTERVAL_1,avg_temp1,[1,2])
    evaluate_models_on_training(INTERVAL_2,avg_temp2,models,city.title())


def annualCityTemp(city, year):
    import calendar
    city = city.upper()
    while city not in CITIES:
        [print(c) for c in CITIES]
        city = input('City not available, select a city from above list: ').upper()
    
    while not min(INTERVAL_1) < year < max(INTERVAL_2):
        year = int(input('Please pick a year between 1960 and 2016: '))
    
    raw_data = Climate('data.csv')
    monthly_mean = [raw_data.get_monthly_mean(city,i,year) for i in range(1,13)]
    months = [calendar.month_abbr[i] for i in range(1,13)]
    x_vals = np.arange(len(months))
    plt.bar(x_vals, monthly_mean, align='center', alpha=0.5)
    plt.xticks(x_vals,months)
    plt.ylabel('Mean Monthly Temperature Deg C')
    plt.title('Monthly Mean Temperature of {} during {}'.format(city.title(),year))
        
