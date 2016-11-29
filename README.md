# NYC-taxi-predictor











NYC Taxi Trips Final Project Report
December 10th, 2015
Team Members - Charu Khatwani, Samridhi Gupta, Sean Richardson

Contents 


Project Purpose
Value Proposition
Assumptions
Constraints
Risks
Project Deliverable
Project Milestones and Work Products
Project Roles and Responsibilities
Approach
Issues Faced
Technologies Used
Explanation
User Interface
Analysis
Future Work


Project Purpose




People waste time and money where they are not able to predict the best time of day when they should make the trip.This will help people to better predict and optimize their trip in a way that saves time and money.




Value Proposition




Making predictions on consumer data after analysis of NYC taxi trip data, helping people make better decision regarding time and fare at some particular time of the year. 




Assumptions


1.Data will generalize across years.

2.Road construction will not skew the data - bridges, road lanes,etc.

3.Public transit system remains same.

4.Speed limit has not changed.

5.Fare changes has not changed the probability of people taking taxi.

6.There are no other taxi services like Uber.

7.Weather patterns will remain same across years, and seasons will occur during the same months.

8.The user pays the toll amounts, and that it will average into a longer distance trip.

9.The times during the same season will have similar traffic.




Constraints


Constraint
Impact
Mitigation Plan
Time Constraint
May be not all data we have can be fully processed
Try and create algorithms using smaller data sets and expand when it’s done.
Human Factor - people seem to go for other options like walk,public transit.
Will skew our predictions because of unpredictability of people.







Risks


Risk
Impact
Mitigation Plan
Data dependability
Predictions will be incorrect
Attempt to generalize and test,validate using a proper validation method.
Functional User Interface
Less user interactive
Make a schedule and adhere to it.




Project Deliverable




Project Deliverable:
Prediction Model 
Acceptance Criteria:
Python source code that generalize predictions



Model that makes predictions based on time of day and seasonal pattern.



Interface that is user friendly in giving us the predictions.


Approach




We first imported raw taxi data from CSV files  from https://storage.googleapis.com/tlc-trip-data/. These data were made available from the Freedom of Information Act, but are now available on the NYC taxi website at http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml into a PostgreSQL using scripts available at https://github.com/toddwschneider/nyc-taxi-data.git 
We processed it to a SQL standard data structures that allowed for faster processing and better ordered data.
This was a time consuming process and took  approximately 5 days to download and import into SQL
Next, we sorted through the data by using simple SQL statements to determine the fields that were appropriate for predicting the data
We divided the data into several tables that represented the seasons. We figured that the season would be an appropriate divisor for predicting the fare.
We further paired down the data, since it was over 850GB by creating temp tables to use for building the classifier.


Issues Faced


We took approximately 5 days to download the data and process it into SQL database.
Space Issue - We predicted that we would have a space issue as it took approximately 400 GB of space to store the data after processing about ⅓ of the dates, but in the end it only took about 880GB of total data for the database.
Calculating distance travelled given only the pickup and dropoff latitude and longitude proves difficult to do. At first we used the geodesic distance (as the crow flies), but this was too far off from actual data to be a good predictor. Then we discovered APIs available online, but using them was difficult and time consuming, but eventually worked.
Predicting the actual number of time that the taxi is sitting in traffic is difficult to do, since there are so many random variables, including accidents, weather, and road closures. This was the essence of the predictor to predict this number.



Technologies Used


Python
Scikit learn - For building the predictor
CherryPy - For running the server to serve the user interface
Psycopg - For connecting to the SQL database
numpy - For quickly formatting the data
matplotlib - For plotting the data for visualization
Other builtins
AngularJS
Twitter Bootstrap
PostgreSQL
Various Linux and Unix shell commands




Explanation


11.1  Choosing the appropriate features


Selecting features that helped us to predict the taxi fare and discarding the irrelevant ones was an important process. The features that we used were the pickup date-time, the distance the taxi will travel based on the pickup location and the dropoff location, the fees associated with the travel taken from the NYC taxi website, and the total number of taxis that were historically picked up during that hour of the day.
We chose the pickup date-time since we figured that the fare would increase during different times of day. For example, during the most peak times, the fare would be higher because of the traffic, additionally, we separated by seasons which gave us additional data, since more people ride during the spring, summer, and fall months and there is more traffic due to high volume and higher construction.
We chose the distance the taxi traveled based on pickup location and drop off location, since that is all the data that we have as input, but the fare is directly proportional to that distance. 
We chose the fees associated with the travel taken, since the fees are available on the NYC taxi website at http://www.nyc.gov/html/tlc/html/passenger/taxicab_rate.shtml and are guaranteed to be included as part of the fare. The only things that are missing are the flat rate fees based on airport location and the time spent not travelling due to traffic. This is the major prediction that our program creates.
We chose the total number of taxis that were historically picked up from a specific hour of the day, since we figured that this might factor into the number of drivers currently on the road. We discovered that there is no additional cost for having higher amounts of pickups during a specific time, but it is a good indicator of “rush hour” traffic and thus a good feature to use in the predictor.
Additionally, we use a sample of data from previous years from the previous season to add additional predicted time. This helps account for seasonal delays such as increased tourism, increased construction, or increased delay due to weather.




To begin, we separated the pickups based on the seasons. This allowed us to make faster predictions given a specific datetime input, but was still generalized enough that the predictor should make sufficient fare predictions. We made the assumption that the seasons were winter between November and February, spring between March and May, summer between June and August, and fall between September and October. We also assumed that the causes of things that alter the fare would be generalized over the years but specific to the seasons. For example, the winter months would have increased traffic due to snow, but decreased number of total pickups and therefore overall traffic.


# We are taking 3 years of data and dividing it into 4 seasons - Winter,Spring,Summer,Fall
       
# Months - December,January and Feburary correspond to Winter
SELECT * INTO trips_winter
FROM trips WHERE EXTRACT(YEAR FROM pickup_datetime) IN (2012,2013,2014) and EXTRACT(MONTH FROM pickup_datetime) IN (12,1,2);
 
# Months - March,April,May correspond to Spring      
SELECT * INTO trips_spring
FROM trips WHERE EXTRACT(YEAR FROM pickup_datetime) IN (2012,2013,2014) and EXTRACT(MONTH FROM pickup_datetime) IN (3,4,5);


# Months - June,July and August correspond to Summer
SELECT * INTO trips_summer
FROM trips WHERE EXTRACT(YEAR FROM pickup_datetime) IN (2012,2013,2014) and EXTRACT(MONTH FROM pickup_datetime) IN (6,7,8);            


# Months - September,October and November correspond to Fall
SELECT * INTO trips_fall
FROM trips WHERE EXTRACT(YEAR FROM pickup_datetime) IN (2012,2013,2014) and EXTRACT(MONTH FROM pickup_datetime) IN (9,10,11);  




Next, we selected queries that only brought in the data that we would actually use in python to limit the time taken to perform the query. The queries that we used to pull data into python were:


queries = {
  'winter': "Select pickup_datetime,dropoff_datetime, pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,trip_distance,fare_amount,tolls_amount from trips_winter",
  'summer': "Select pickup_datetime,dropoff_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,trip_distance,fare_amount,tolls_amount from trips_summer",
  'fall': "Select pickup_datetime,dropoff_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,trip_distance,fare_amount,tolls_amount from trips_fall",
  'spring': "Select pickup_datetime,dropoff_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,trip_distance,fare_amount,tolls_amount from trips_spring"
  }


Additionally, we created a helper function to assist in pulling in the data to the python predictor program. The function allowed us to use LIMIT to limit the total number of queries and to use tables that were appended with _temp with a limited subset of the data to allow for more streamlined testing:


def query_sql_server(season, limit=None, test=False):
  conn = psycopg2.connect('dbname=nyc-taxi-data user=khatwacu')
  cur = conn.cursor()
 
  query = queries[season]
  if test:
      query+="_temp"
  if limit:
      limit = " LIMIT "+str(limit)+";"
      query += limit
  else:
      query += ";"
  cur.execute(query)
  result = cur.fetchall()
  return result


Next, we calculated the total number of pickups per hour based on the season. The times that were most active were 1800hrs (6pm) during spring and fall months and 1900hrs (7 pm) during winter and summer months. We used a sql query generate these based on seasons:


SELECT EXTRACT(HOUR FROM pickup_datetime) as hour ,count(*) from trips_winter group by  EXTRACT(HOUR FROM pickup_datetime) ;
SELECT EXTRACT(HOUR FROM pickup_datetime) as hour ,count(*) from trips_summer group by  EXTRACT(HOUR FROM pickup_datetime) ;
SELECT EXTRACT(HOUR FROM pickup_datetime) as hour ,count(*) from trips_fall group by  EXTRACT(HOUR FROM pickup_datetime) ;
SELECT EXTRACT(HOUR FROM pickup_datetime) as hour ,count(*) from trips_spring group by  EXTRACT(HOUR FROM pickup_datetime);







plt.xlabel('Hour - Red(Winter) Blue(Summer) Yellow(Spring) Green(Fall)')
plt.ylabel('Number of Taxi Pick ups')
plt.title('Taxi Pickups per hour')
plt.grid(True)


x= xrange(24)
 
We included the output from these in the python code below:


# Total number of cab rides separated by hour and season. So we can see when there is a surge and increase accordingly
winter_totals = [483170, 377465, 289465, 235185, 202440, 125305, 141890, 285220, 431830, 449615, 438610, 442790, 461310, 482780, 549985, 626395, 703030, 762850, 820945, 807780, 750410, 683165, 632505, 569580]
summer_totals = [974705, 777635, 582415, 431140, 365000, 243110, 274255, 448055, 697000, 750935, 711995, 713545, 733635, 749185, 856475, 964925, 1049425, 1134990, 1267410, 1260415, 1124705, 1123205, 1119135, 1096340]
spring_totals = [893665, 698845, 518290, 419890, 350450, 215560, 251735, 490785, 734820, 759130, 721815, 715505, 734235, 756660, 867665, 976930, 1076620, 1201875, 1314915, 1332795, 1269700, 1210250, 1162950, 1050370]
fall_totals = [972995, 798080, 600835, 468365, 385880, 246225, 307965, 587575, 838155, 860070, 811170, 799905, 813680, 838900, 955660, 1061620, 1162770, 1300880, 1422010, 1443415, 1355205, 1269630, 1230810, 1149250]
season_totals = {'winter':winter_totals, 'summer':summer_totals, 'spring':spring_totals, 'fall':fall_totals}


plt.plot(x, winter_totals, 'r--', x, summer_totals, 'b--', x,spring_totals, 'y--',x,fall_totals,'g--')
plt.show()


Next, we formatted the data as a dictionary of features then used the scikit-learn DictVectorizer module to import the data in a format that scikit can use. Once the data was formatted, we used different classifiers to test to see which features and classifiers would work best. 
This was a long process involving trial and error to determine which features to throw out and which to build upon. We used a K Fold technique with 10 folds to test to see if the data would match and positively predict the fare. This meant that the data was split up randomly into 10 parts each of testing and training data pairs. 
At first, all the features that we could think of were included and all the classifiers that we could think of. This was done on a small subset of the data to allow for quick refactoring. The problem with using all the features, was that the prediction rate for nearly all of the classifiers was very poor. Once we limited the number of classifiers, this allowed for a better prediction rate. 
Finally, we kept increasing the amount of data to test against and further paired down the features to use while predicting. At the end, we saw a 100% prediction rate using K Fold to make sure the predictor was generalized across all the data that we used. The predictor with the best classification rate went between Decision Tree Classifier and Decision Tree Regressor, with the regressor having occasionally wild predictions. We chose, then to rely on the Decision Tree Classifier as the classifier for the predictor.


from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.naive_bayes import GaussianNB as GNB


def run_cv(X,y,clf_class,**kwargs):
  X_old = list(X)
  y_old = list(y)
  X = np.array(X)
  y = np.array(y)
  # Construct a kfolds object
  kf = KFold(len(y),n_folds=10,shuffle=True)
  y_pred = y.copy()
  clf = None
  classifiers = []


  # Iterate through folds
  for train_index, test_index in kf:
      X_train, X_test = X[train_index], X[test_index]
      y_test = y[test_index]
      y_train = y[train_index]
      # Initialize a classifier with key word arguments
      clf = clf_class(**kwargs)
      clf.fit(X_train,y_train)
      if 'feature_importances_' in dir(clf):
          print('{}'.format(clf.feature_importances_))
      y_pred[test_index] = clf.predict(X_test)


      classifiers.append(clf)


  return y_pred,classifiers


def accuracy(y_true,y_pred):
  # NumPy interprets True and False as 1. and 0.


  return np.mean(y_true == y_pred)




def test_features(date=None):
  if not date:
  date = datetime.datetime(2014, 12, 1, 2, 3, 4)
  season = make_season(date)
  totals = season_totals[season]
  response = query_sql_server(season, None, True)


  #input latitude and longitude pickup
  #input date and time
  response_l,y = make_season_measurements(response, totals)
  X = transform_meas(response_l)


  classifiers = []


  print "Feature space holds %d observations and %d features" % X.shape
  print "Unique target labels:", np.unique(y)


  print "Support vector machines:"
  acc, clf = run_cv(X,y,SVC)
  classifiers.append(clf)
  print "%.3f" % accuracy(y, acc)
  print "Random forest:"
  acc, clf = run_cv(X,y,RF)
  classifiers.append(clf)
  print "%.3f" % accuracy(y, acc)
  print "K-nearest-neighbors:"
  acc, clf = run_cv(X,y,KNN)
  classifiers.append(clf)
  print "%.3f" % accuracy(y, acc)
  print "Decision Tree Classifier:"
  acc, clf = run_cv(X,y,DTC)
  classifiers.append(clf)
  print "%.3f" % accuracy(y, acc)
  print "Gaussian Naive Bayes:"
  acc, clf = run_cv(X,y,GNB)
  classifiers.append(clf)
  print "%.3f" % accuracy(y, acc)
  print "Decision Tree Regressor:"
  acc, clf = run_cv(X,y,DTR)
  classifiers.append(clf)
  print "%.3f" % accuracy(y, acc)
  y_test = clf.predict(X)
  matrix = metrics.confusion_matrix(y_test, y)
  score = clf.score(X, y)


  print('accuracy: {}'.format(score.mean()))
  print(matrix)


  return classifier




User Interface


Once we recognized the features and created our model, our next challenge was to make our utility user friendly. Features we had chosen to make predictions are not intuitive enough. Not many people know about the longitude and latitude values of their origin and destination point of their journey. Hence to help them and communicate with them in human understood language, we created an user interface where people can enter their pickup and dropoff location, date and time of the day to know the estimated fare.






Web Interface Approach:
Twitter bootstrap and Google AngularJS were used to generate the frontend. Google Map API is used to convert an address to geoLocation latitudes and longitudes and show on a Map. Web application is built with backend set in python with exposed REST APIs. 


var map = null;
var geocoder = null;

function initialize() {

    map = new google.maps.Map(
            document.getElementById('map_canvas'), {
center: new google.maps.LatLng(40.4419, -73.1419),
zoom: 5,
mapTypeId: google.maps.MapTypeId.ROADMAP
});
geocoder = new google.maps.Geocoder();
}






JavaScript:
var app = angular.module('nycTaxiApp',[]);

app.factory('DataService',["$http", function ($http) {
    var fare = null;
    return{
        predict: function (data) {
            var url = "/predictor";
            $http.post(url, data).success(function (prediction) {
            fare = prediction;
                console.log(prediction);
            });
        },
            getPrediction: function () {
                               return fare;
                                       }
    }
}]);

app.controller('dataController', function ($scope,DataService) {

    $scope.pickup= "Times Square"; 
    $scope.dropOff= "JFK airport";

    $scope.date ="07/07/16";
    $scope.time ="09:20:40";
    $scope.predict = function () {
        data = {
            "pickupLongitude" : $scope.pickupLongitude, 
    "pickupLatitude" : $scope.pickupLatitude,
    "dropOffLongitude": $scope.dropOffLongitude,
    "dropOffLatitude": $scope.dropOffLatitude,
    "date" :$scope.date,
    "time":$scope.time,
    "distance":$scope.distance
        }   
        getDistance();
        DataService.predict(data);
    };


    $scope.predictedFare = function () {
            return DataService.getPrediction();
    };


    var getDistance = function () {
        var directionsService = new google.maps.DirectionsService;
        var directionsDisplay = new google.maps.DirectionsRenderer({map: map});

        directionsService.route({
            origin: $scope.pickup,
            destination: $scope.dropOff,
            travelMode: google.maps.TravelMode.DRIVING
        }, function(a, c) {
            if (c == google.maps.DirectionsStatus.OK) {
                directionsDisplay.setDirections(a);
                for (var b = 0, f = a.routes[0].legs, d = 0; d < f.length; ++d) b += f[d].distance.value;
                var f = b / 1E3,
            d = 6.21371192E-4 *
            b,
            h = 5280 * d;
        console.log("Driving distance: " + d.toFixed(2) + " miles");
        $scope.distance = d.toFixed(2);
        console.log( $scope.distance )

            } else {
                window.alert('Directions request failed due to ' + status);
            }
        });
    } 


    var getLatLng = function(address, setMarker){
        geocoder.geocode( { 'address': address}, function(results, status) {
            if (status == google.maps.GeocoderStatus.OK) {
                map.setCenter(results[0].geometry.location);
                console.log(results[0].geometry.location);
                console.log(results[0]);
                var marker = new google.maps.Marker({
                    map: map, 
                    draggable: true,
                    position: results[0].geometry.location,
                    address: results[0].formatted_address
                });
                map.setZoom(10);
                markers.push(marker);
                setMarker(marker.position);
                marker.addListener('click', function() {
                    setMarker(marker.position);
                    console.log(marker);
                });
                marker.addListener('dragend', function() {
                    setMarker(marker.position);
                    console.log(marker);
                    console.log(marker.position);
                });
            } else {
                alert("Geocode was not successful for the following reason: " + status);
            }  
        });
    }

    var markers = [];
    var pickupSetter = function(pickupPoint){
        $scope.pickupLongitude = pickupPoint.lng();
        console.log(pickupPoint.lng());
        $scope.pickupLatitude = pickupPoint.lat();
    }

    var dropOffSetter = function(dropOffPoint){

        $scope.dropOffLongitude = dropOffPoint.lng();
        $scope.dropOffLatitude = dropOffPoint.lat();
    }


    var markers = [];
    $scope.showAddress = function () {
        for (var i=0; i<markers.length; i++) {
            markers[i].setMap(null);
        }
        getLatLng($scope.pickup, pickupSetter);
        getLatLng($scope.dropOff, dropOffSetter);
    };

});


Python Server Side:
import os, os.path
import random
import string
import json
import datetime
from predictor import predict_fare

import cherrypy

class Predictor(object):
    @cherrypy.expose
    def index(self):
        return open('index.html')

class PredictorWebService(object):
    exposed = True

    def POST(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))

        jsonObj = json.loads(rawbody)
        print jsonObj
        distance = jsonObj['distance']
        day = int(jsonObj['date'][:2])
        month = int(jsonObj['date'][3:5])
        year = int(jsonObj['date'][6:8])
        hour = int(jsonObj['time'][:2])
        minute = int(jsonObj['time'][3:5])
        second = int(jsonObj['time'][6:8])
        date = datetime.datetime(year, month, day, hour, minute, second)
        p_latitude = jsonObj['pickupLatitude']
        p_longitude = jsonObj['pickupLongitude']
        d_latitude = jsonObj['dropOffLatitude']
        d_longitude = jsonObj['dropOffLongitude']
        fare = predict_fare(date, p_latitude, p_longitude, d_latitude, d_longitude, distance)
        return str(fare)

if __name__ == '__main__':
    conf = {
            '/': {
                'tools.sessions.on': True,
                'tools.staticdir.root': os.path.abspath(os.getcwd())
                },
            '/predictor': {
                'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
                'tools.response_headers.on': True,
                'tools.response_headers.headers': [('Content-Type', 'text/plain')],
                },
            '/static': {
                'tools.staticdir.on': True,
                'tools.staticdir.dir': './public'
                }
            }
    webapp = Predictor()
    webapp.predictor = PredictorWebService()
    cherrypy.server.socket_host = '0.0.0.0'
    cherrypy.quickstart(webapp, '/', conf)



Analysis




We analysed fare for different season and then over different time of the day.
We saw that during daytime (6 AM) onwards there was an increase in the fare. It  was maximum at the peak traffic time between 6 and 7 PM, then it started to decrease  and decreased to the lowest point during night time. This trend of the fare was found to be true for each season. 
Fare trend showed that during summer fare was the highest and it decreased as winter approached.








Fare for summer months for different times of day:


Night time: 3:20 am	Fare: 47.65$



Day time: 6:20 am	Fare: 49.21$







Day time: 9:20 am	Fare: 53.79$















Day time: 10:20 am	Fare: 54.16$





Day time: 12:20 am	Fare: 54.16$



Here we see an upward trend in the fare, it increase as the day proceeds, then becomes constant and then decreases with night time.




Fare for fall months for different times of day:


Night time: 3:20 am	Fare: 45.5$





Day time: 9:20 am	Fare: 46.0$



Fare for Winter months for different times of day:


Night time: 3:20 am	Fare: 38.46$









Day time: 9:20 am	Fare: 41.9$



Day time: 12:20 am	Fare: 41.9$



When comparing fare predicted for same time across different season, we see an decreasing trend in the fare. Fare are highest in the summer and they decrease as the winter approaches. 







Future Work


Predict other things such as peak traffic times and rush hour times; possibly predict the tip that a fare will give based on the location; predict the best time to travel based on starting and ending points..
Detect when a dropoff or pickup location is in a flat rate location or in a location that charges additional fees such as a bridge, tunnel or toll or in Newark.
Create heatmap on the user interface to see the fare trend over long duration of time.  






