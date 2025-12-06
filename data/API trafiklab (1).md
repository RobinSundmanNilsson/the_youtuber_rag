# API trafiklab

[00:00:00] Hello and welcome to this video where we'll go into getting data from an API. And the API that we've chosen is Trafiklab. And from this API, you will be able to get data on public transport. 

It's good to, understand a little bit about the data set so that you could, for example, monitor if there's delays in the trams or trains. So yes moving on, we'll go into the web browser directly now.

So here I'm in the web browser and I've gone into Trafiklab.

se slash API. Let me move myself here.

In Traffic Lab SC slash api~~ we have ~~we have several APIs that we can work with. And~~ one, ~~the ones that we will pick are those that are in s robot. So here race robot stalled. It's tabella here. You can get the timetables for different stops.

So we'll go in and ~~see ~~see more details, how to work with this one. [00:01:00] And this re robot plan is used for ~~planning~~ planning your trip. For example, you want to travel ~~from Sweden to ~~from UBO to Stockholm, ~~for example. ~~Ubo to Malmo. You can find out. ~~The trips~~ which type of trains and buses there are and their stops, et cetera.

So this is quite interesting and it's an open API so that you can use it for free. The only thing you need to do is ~~to ~~to create an account and register ~~for ~~for these APIs to use. Firstly. ~~You, ~~if you scroll up here, there's Minna sidor or my pages. So clicking on this one we'll go in here and you can create an account for me, I will just log in with GitHub so that you can do if you have a GitHub.

And here~~ I have ~~you should create your own projects. So basically I have a few projects, I have streaming data and I have this testing and there's also a sandbox here. So you should create a new project.[00:02:00] 

So when creating a new project, you give it a name. I will give it a, for example, a travel planner, ~~for example, ~~and ~~you can have a homepage if ~~you can link to a homepage if you want, but if you don't, it's okay. You can choose what type of project it is. It's an educational project and open source repository if you have that.

But~~ unless you don't need, ~~you don't need to. ~~And ~~description is also good to have or it's mandatory because these are optional. Description learning. Let's see. Learning. Lab a p. Same.

So here ~~I have~~ I have the travel planner ~~and ~~and now you should get your API keys ~~and to get the ~~. It's basically like this. ~~You choose here~~ you choose which type of API you want. And the one we want to pick is a robot

me see here. Race robot version 2. 1. And then you click add key to project and you will get an [00:03:00] API key. 

Here are the API keys. I won't go down to the API keys because I don't want to share them with you. Instead you should go down there and copy ~~that key~~ that specific key because we will use it ~~in order to get ~~in order to use this page~~ get the ~~APIs.

So moving on to when you have the key store it in a dot ENV file, I'll show you how that looks soon. Or actually we can go there directly. I'll go to Visual Studio Code. And here~~ in this~~ in this project, I'll open a new terminal and let's see what I will do. I will do touch dot ENV.

So that we get a dot ENV file and inside this dot ENV file, you should write API equals to your API key the ones that you've copied. So just copy and paste it here and then we'll go back to Visual Studio. And we'll go back to. Traffic [00:04:00] lab and inside of API keys or actually documentation.

You've got the documentation and you're going to API overview, right here, and we can go into this robot started stability, for example, click on this one and here there is a example on how you call this one. So you basically just copy this one. Departures and arrivals, right? So you copy this one in order ~~to ~~to get the data from this one.

And, but then you need to create an if you're using Python, which we're using in this course, and in this lecture we will~~ we'll ~~use the request library and we will do a get request to this end point and using our API key. And also you can go down here to read a little bit more about different options and what they mean.

So I won't go through them instead I will leave it to you for you to read. But let's go back to Visual Studio Code now [00:05:00] and let's set up different things. So let me do this. I will create the virtual environment. So uvvnv source dot. So now I've activated my virtual environment and I will do uv pipinstall.

What do I need? I need ipykernel, of course, because I will work with Jupyter Notebook. I also need pandas. And I also need And let me think, what do I need requests? I need yes, starting with this. I think not dot ENV of course. So Python dot ENV. One thing with env it is for so that you can store your secrets.

I'll show you what that means soon. So let's create a Jupyter notebook file now. So touch

let me see touch [00:06:00] API, public transport, ipnb. Okay. And now we'll open this one.

So we'll do an EDA on public transport traffic lab API. And then I'll change this to my virtual environment.

So let's start with testing out our dot ENV. ~~So from dot, let me change this to Python and I'll increase it, making it a little bit bigger. ~~So from dot ENV import load dot ENV. And we need to import OS as well in order for us to get the environment variable and then we'll do import requests .

And starting with our load. dnv, we can do like this. Start with load. dnv. In general, you need to have a path here. But~~ in, ~~when you're in Jupyter Notebook, you have the path is based on where ~~this ~~this file is so since e NV is a sibling [00:07:00] to this file and this Jupiter notebook, ~~then~~ then do ENV will be found.

So when we do low e nv and then we can do os get ENV, and we take the API, if we are going to nv, we can see that it's called, it's just called. I'll call it API key. And then I will do API key here. And if I run this one, we can see we get dot. That means that ~~we~~ the Jupyter Notebook finds this dot DNV.

Okay, great. So now I'll just pause the video and I'll put in my real key. So remember, the real key can be found ~~in ~~in strategic lab where you're logged in.

I've added my real key and of course I won't show that to you. It's important that you add do ENV ~~to your ~~to your.gi nor file. If it isn't there already, so that is to [00:08:00] make sure that it's not tracked by Git and that you don't accidentally push it to a public GitHub repository.

So this is in order for us to so because we don't want to write out the API key here. Directly because when I push it~~ you ~~then people will see it ~~in my directory, ~~in my repository and can use my API key. So secrets store it in DNV and load it using Load d nv. Okay, so then this one, I can just call it API Key equals to this.

So if I run this one it, it is stored in API key, but~~ I ~~I don't write it out so that you can't see what it is. However, I will use it later on. So now we'll go into Trafiklab. So here in Trafiklab, let me see. So in traffic lab we have if we go into our ~~this robot~~ [00:09:00] this robot route planner, you can start here and you can see this example call.

Let's copy this one. So it says. This example called ~~receive ~~retrieves all routes from Stockholm Central Station ~~seven~~ this ID to Malmö Central Station, another ID. So I just copy this one and I go back to my Visual Studio code and I will call URL equals to this, make it into a string and make it into an F string.

And then I will also make this API key into a variable so that I can use this. API key inside of this URL. Okay so since we have, we can read it out here. So api. resrobot. se version 2. 1 slash trip. So ~~this is the end, so it's~~ it's the API. We query, we're using the format. This is an [00:10:00] option. The option is a format which is JSON.

And another option, origin ID equals to this ID. And dest ID is destination ID is this ID here. And then we have pass list equals true and show passing points equals true. Actually, I'm not sure what both of them are. ~~It's ~~it's listed in the documentation in the web page, which you can read about.

And then most important, very important, is that we have the access ID, which is our pass list. API key, right? So that is very important. And we get the API key from our dnv. So that is how everything is connected. So now we have our URL. So this is an endpoint for us to retrieve data. So in order for us to get the data we need to do requests dot get, and basically we just put in the URL.

And we call [00:11:00] this response equals to this response. And I run this one response 401. Why is that? Let me pause and I will pause and debug this.

Okay, I found a problem. First of all, I wrote response. txt to see~~ to see what~~ what text we get. And we get an error code, which says API auth, error text access denied. Okay, then I thought based on this, what could be the problem? Maybe the API key is wrong. So what did, ~~I just, ~~I typed out API key and saw what I get.

Dot dot. How strange, even if I re ran this one, I got API key. Because this is already loaded, right? That is how it works in Jupyter Notebook. When it loads one time, it won't load again. So this is important if you have~~ if you ~~started with something else than ~~your API, ~~your real API key. Just do restart.

~~Restart this one, ~~and now I will [00:12:00] remove this cell here. And if I run this one, now it should work. Yes, now it works. We can see there's a response. txt, and we have Stockholm Central Station and a lot of things here. So instead of getting it as a text, I want to get it as a JSON object. ~~So as a JSON, ~~so JSON object is basically in Python is represented as a dictionary.

So you have a dictionary of a list and you have a lot of other things. So in dictionaries, a very important thing what I usually do when I start working with API where I get JSON data is that I usually Try to explore this API, explore this dictionary. So starting with this, let's continue.

We can do like this. Let's see. Response. json we can get let me see. Response. json, we can call it results,[00:13:00] 

response. json, result, and we do result. keys, and we can see what type of keys we have. So we have trip, we have result status, and technical messages, server version, et cetera. So there's a lot of things here. And actually in my Jupyter notebook~~ in the. ~~In the lecture note, you can see that I've explored most of them, I think.

However, ~~you should also ~~you should also check them out. What we can do is like this. Since it's a dictionary, we can take out the trip here. And we can see that it's a list. It starts with brackets here. If it's a list, we can do like this. We can do length of this. And we can see seven.

Okay, there was seven here. We could do a result of result status and see what that is. Okay. ~~Some time diff ~~some kind of metadata. I'm not sure what this, ~~what it ~~is, but ~~you should, ~~you can read in the [00:14:00] documentation in case you wonder. We can continue here technical messages, and see what we get here.

Okay, so technical messages. We get ~~some, ~~something that I don't understand either. Let's continue going on down here and we'll take ~~serve ~~server version. Probably I can understand what that is. Dialect version plan R T S. I think those are not so relevant for us. So we will continue with the trip.

So results of trip. So you can see here we have it's a list and now it's seven items, right? So what we can do with this is that we could ~~loop through it. So if it's a list, we can ~~loop through it. We can loop through the list. And actually in the lecture notes, I have created A function~~ of this ~~of the get request, basically.

So that is a little bit [00:15:00] more elaborate and ~~you can~~ you can choose it to use. ~~And we can ~~however, here ~~I will I will not write, or ~~I can write out the function, actually. Because we will use it later on. Let's do that. Let's go into the function. I will create a function of it. I will just copy from the lecture note and I will explain that.

This is what I did for the function. We can remove the API key as we had it earlier. So basically it's the same as before. We have a URL, however, the origin ID and destination ID I put it into as parameters so that you can send them in as arguments ~~in order ~~if you want to Choose other destinations and origins.

Now, the first one is Stockholm and the other one is Malmö, but I want to get ~~the ~~just the body later on. So then afterwards we do a try ~~because how come? ~~Because the response object, the request. get, it can fail, right? It can fail with different [00:16:00] exceptions. And if it has an exception we also do response.

raise for status. So in case It'll raise an HTP error if it occurred. ~~Four, 400 x four xx and five xx. ~~So if you get status code 400 something, 500 something, then it's some kind of HTTP error. And that is what ~~I~~ we're catching here with the accept. So this one. Result equals getTrips and we can get result.

keys. And this is basically the same ~~as~~ as I showed before. So now we have the trip, we get result of trip, right? So we can call this, let me see, we can call this we can call, for example exampleTrip equals to this one, and I will take the zero of it. Example trip of, so if I do example trip now, you can see that we get the first one.

So origin [00:17:00] is in Stockholm and destination is in Göteborg, central Stockholm. Interesting. Yeah, I changed also here, this is another ID than the one I used before. So this one, this idea is for Göteborg, but how do we know which idea corresponds to which city or which stop? That, that will come back to later on in in the next section here, but we have Göteborg and Stockholm.

So let's continue. We can we can figure out a little bit more things here. Here we can go into we can go in to example trip of for example we can go into distinct origin, and we can see that we get Stockholm and and it has these keys and these values. We can continue copy this one and go into example trip of destination.

And we can see that, okay, Göteborg Centralstation, et cetera, cool. [00:18:00] And you can explore more things here, but something I thought was quite interesting was so if you go into exampletrip.

~~key, exampletrip, ~~you can run this one. And you can see here, origin, destination, and we have service days. Actually, we can do exampleTrip. keys to find out what we have. So origin, destination, service days, legList is quite interesting. So I'll do legList. And this one, here we can see a lot of information. We have origin, we have destination.

So we have several destinations, it seems. ~~So it seems to be, ~~so we have Stockholm here, and then we have Karlstad, so we can continue to find out different types of stops. ~~Leg list, and then we take out leg. ~~[00:19:00] Actually, ~~this ~~these names, I'm not sure what they stand for, but~~ they are in, ~~this is how they coded the response object.

So we get the list here. This is quite interesting. So let's do length of this list. I want to see how many there are here. Okay. Two. I'll copy this one.

Okay. For example, I will pick the zero

and in the zero, we have origin destination. I can do dot he's. And we can see we have origin, destination, journey status, stop. It's quite interesting. So I'll continue, copy this one. I'll go [00:20:00] into stops. And we can see, okay, stops. We have some key called stop. And then there's a list here. Okay, so we'll pick that one, stop.

And we can see there's a list. Stockholm, Katrineholm, Hallsberg. Okay, see, ~~this is, ~~this requires some domain knowledge. ~~If you're ~~if you have traveled trains between Stockholm and Göteborg. These are common stations that exist there. So these are ~~the common stations or common, ~~the common stops, ~~though.~~

Okay, so this is quite interesting. So we could continue here and we could actually call for example, these are different stops, right? We could call it example stops, example, stops like this. And then I will loop through this one. So I will do like this. For stop [00:21:00] in example stops, and I can do stop dot get name.

So basically get the same as using bracket notation. However, it won't return an error if the key is missing. Instead, it will give you none if the key is missing. So colon stop dot get for example, I will get the arrival time

or departure arrival time. Yeah, I could take departure time as well. Okay, let's see. Yeah, I need curly braces because I want to make a dictionary here. So basically I'm doing a list comprehension of the dictionary. So here we can see, okay, we started with Stockholm Central Station. We go into Katrineholm, Hallsberg, Degerfors, Kristinehamn and Karlstad.

~~Cool. ~~Cool. Maybe the other section, [00:22:00] because the length of these stops were two, right? And this ~~example stop that we, ~~example stops that we have, maybe this is half of the journey. I'm not sure this is worth finding out. Okay. But how did we get the body ~~as~~ as the stop as this as the ID, right?

So this, we need to use ~~some, ~~another API. Called ResRobotStopLookup, so I will go ~~into here, ~~into the browser, and we can see in the browser, we have a stop lookup, and ~~the stop lookup, ~~it will basically, ~~so ~~you have here, this is the API, you have location. name, question mark, input equals just a body, And format equals ~~Jason.~~

And if you do ~~it, ~~the bar question mark, then you will, it's a ~~fussy ~~search so that it will match something that is similar to it. So basically here, you can see ~~what they said, like ~~the input ~~here, input as a, ~~as the option and [00:23:00] the search string, append the question mark for ~~fussy ~~search, that means we will find something ~~if we will find ~~for example, if we take.

~~If we just take, ~~without the question mark, we'll only get Göteborg here. This is the data we get. We get the ID, external ID, name, ~~and ~~longitude, latitude, etc. But with Fuzzy Search, ~~we'll get everything that, ~~we'll get things that are similar. ~~So we get Göteborg etc. ~~So everything with Göteborg is matched here.

Okay, cool. But then we find the external ID and then you use the external ID to put it into ~~your ~~your travel planner or your route planner or your ~~time ~~timetables. So moving back to Visual Studio Code.

Okay so let's go into the stop lookup API. Here, we have stop look up API. We use this to find the ID. I will copy the URL. And basically I can copy all of this [00:24:00] here. I'll change this to Python. And then, okay, if you look into What we did here. So location is just the body. I have an F string so that I search and get the body.

And this is a, an exact search. Now we don't have a question Mark and we have a API key and we have a result basically. So if I run this one,

we get the stop location or cord location,

and we can do results let's see results of. Stop stop location or cord location. And I run this one and we can see here we have stop location

and you can see name is yet the body

and we can see, let's see, this is a list, right? So I can [00:25:00] do length of this list and it it contains 10 values. So basically, okay, we have 10 10 stops. Okay, so I will do result of, actually I will call this one, I'll remove this one like here, and I'll do stop locations equals to this, give it a variable, stop locations, and then I'll do stop locations, I'll run this one.

So we have for example, zero, stopLocation zero, and we could take stopLocation zero of stopLocation, and we can dot, take dot keys, and we can see the keys here. Okay, so it has an external ID, and it has a name. That is interesting. So how, what if I do like this? I copy this one, and I take out [00:26:00] name.

Cool. And if I copied this one and ~~I take out ~~I take out X ID, external ID, and we get, The external ID. So perfect. So this is how I got the ID for in order to put it into my route planner before. But if we have several, we want to pick the one that fits the best for us. Then basically ~~just ~~just do like this.

We can loop through this. Let's do four, stop location, in stop locations. Remember that stop locations is a list. Stop locations is a list of ten values. And inside of stop locations there is, so when I look through it, I will get This value and then I will get 0, 1, 2, 3, 4, up to 10 or up to nine.

So [00:27:00] here, this is a for loop and I want to have ~~stop location or stop equals ~~stop location or stop location, right?

And why do I do that? We can see that I did it here because that is how it was structured. So I needed to take this. guy and do get the stop location. And from the stop location, I need to find the name, right? So then I can print the stop of name and stop of external ID. So here you can see, here are all the IDs for Göteborg.

So basically I can also do, I will do a little bit better formatting here. So print fString, stop,[00:28:00] 

stop like this, and it should have a left alignment of 50 character wide field. This is what it means using fString formatting. And then afterwards, after these 50 characters, I want to have My next string. So my next string is external ID. So then I get stop here and I get the external ID here. And then I could left align these also.

So that it comes into the external ID. So what I do is here stop. I need to do similar here. Like this, and I need to do it into F string like this,

like this. If I run this [00:29:00] one, let's see. Yes, nothing happens, but if I do colon 50 character wide, and this means left alignment, other side means right alignment. So then, perfectly, now it's now it's easy for me to find out ~~which~~ which one I want. So if I want to have, for example Volkswagen, then I just pick this ID here.

Okay, great. And, yes, I've actually in my lecture note you should look into that. I have put it into a function which works. Which you can use for finding other locations that is actually quite good to have us as well. So I will copy that one. So this is from my lecture note, access ID from location.

So you just type in the location and it will find bind it for you. And basically it's the same idea as before, however, ~~one important, ~~some important things here is that I use [00:30:00] so this is quite interesting. Okay, so this you have seen before you get the response object. And you get it as a JSON, you put it into try because there can be exceptions.

Then you do this for loop. And here is one important part, is that stopData equals to next of iter of stop. values.



That is the logic. And then it has an except for for some errors. But with this one, you can just use this function immediately. So access ID from location. We can, for example, get Malmö, and we're doing a fuzzy search so that we have here, we see Malmö, and you can get the external ID, Malmö, here, another ID.

~~You can~~ you can, for example, find a very small city here called Unsala. And you can see, yeah here's what's found in [00:31:00] Onsala. Actually, there's more stops in Onsala, but they don't have external IDs. And that's why ~~so ~~you can actually test it out without this if statement here. And ~~you will see ~~you will see more stations that corresponds to Unsala but~~ that, ~~that's why ~~I use this ~~I use this one in order for me to stop so that I can get the external ID because without the external ID, I cannot use it further anyway.

And also ~~I will for example, ~~I can find something else. This interesting is another small city. And you can see, yeah, there's a lot here. Okay. ~~But ~~perfect. Now ~~we have~~ we have this and we can move on to another API. We can move on to time tables. So in time tables, we go into this robot and we have time tables here.

And similarly, just copy this one here for departures and one for arrival. [00:32:00] You can pick the ones you want. ~~And let me. ~~Let me go up a little bit and find will actually go and copy. I ~~will ~~copy from my lecture note here. So this is for Korsvagen stop ID, because if I go up here, I have Göteborg Korsvagen is this ID.

I will copy this one.

And I go in here and store it. So this is the stop ID. I use this one to get the results I want. ~~And so you see it's another API endpoint. ~~And now actually I won't in this part, I won't put it into function, but~~ it's a, ~~it is good practice for you to put it into function ~~and ~~and do that and I will leave that for you to do.

Instead I will just pick out some, I will show you using a data frame. Now ~~the ~~import pandas as pd, [00:33:00] we could do like this. If we write out results. Results is a JSON object, right? That looks like this departure and it has a lot of other things. However, I'll just pick out the departure.

I'll actually throw this away because it will clutter my notebook. ~~When ~~otherwise I will do ~~the~~ the PD dot data frame of results of departure. That is where my data is. So df equals to this, df timetable, df timetable ~~dot~~ dot head, right? Head, like this, and you can see okay, so this is quite neat to work with.

If you have a data frame, it's much neater than working ~~with~~ with the ~~Jason ~~objects directly.[00:34:00] 

So before we unnested the ~~Jason ~~objects and we looped through to get the things that we wanted, but now you can use the power of pandas and your panda skills in order to data process this. So basically you could do like this. dftimetable. columns for example. And we can see what ~~the ~~columns we have.

~~We could do dftimetable ~~We can pick out ~~we can, ~~for example, clean it so that for example, we want to pick out what do we want? We want name and we can see, yeah, ~~this is, ~~these are the names we could pick out. What do we want more? We could pick out the stop and we have all the stops here.

We could pick out longitude, latitude, which you could use to draw in a map, dots in a map if you want. And see if you can map out the body direction [00:35:00] here, you can see where the directions are for the ~~trams on ~~trams and buses, we could ~~do ~~also ~~interesting in ~~which date we are using.

So date okay. Six is today and we can see the time and yeah, I just accessed the data. So you can see that Yeah. Okay. Here is a time in around one hour, one hour time, and you can see that we have, okay, ~~one~~ 116 ~~buses and tra ~~buses and trams included~~ that ~~that the departures ~~from ~~from Kosh wagon.

Okay. So we could call this df timetable clean, for example. And then we could do DF timetable clean. And we could find out for example, name dot value counts. And we can see, yes, ~~these are the unique ones in ~~these are the unique buses and trams ~~in in ~~this station at this~~ station.~~

Particular time frame from ~~from seven, ~~1911 to [00:36:00] 2011. And you can see, yeah, these are the most common ones and yes, ~~they~~ you have uncommon ones ~~are ~~express bus. They are ~~come, they're flying or they're. ~~Going further away and fleet transfer means flight transfer. So they're also more uncommon?

Yes. Okay. And with this~~ you, ~~you can do more exploratory data analysis, but I will leave that for you. So with this, I would like to thank you for watching this video and see you in the next one. Bye.

