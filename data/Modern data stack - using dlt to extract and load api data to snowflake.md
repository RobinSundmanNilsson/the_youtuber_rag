# Modern data stack - using dlt to extract and load api data to snowflake

**Kokchun Giang:** [00:00:00] Hello and welcome to this video where we'll use DLT to extract and load API data into Snowflake. This will require us to make different roles we'll make a user we'll create. That we give ~~the grants ~~the proper grants and privileges to our roles and user. And then we will take a look into the API ~~we'll in investigate and ~~doing some exploratory data analysis on the API ~~and then afterwards ~~when, we have done that, we will go into loading this API data into Snowflake, and then we'll go into Snowflake through Snow SQL and also in s site to check out the data. This is still extract and load part ~~of ~~of the ELT, ~~EL part. And then ~~afterwards we have ~~date. ~~DBT for transformations. We'll go [00:01:00] into the browser first.

**Kokchun Giang-1:** Here I am in the code repository and we'll work with~~ abit for mailing data. ~~A mailing, API, this is a Swedish data for job ads. And we'll go in there and take a look. This is the page. ~~We'll go in there. ~~However I'll also note here before I jump out is that ~~we have this ~~we need to use ~~this.~~

The DLT the secrets tool to set up the connection to Snowflake. We will go through that as well. Remember to note this down 'cause this we'll use later. And then I will go into this page here. ~~We come to ~~let me close this. Code, examples, start here. ~~We have ~~here we have job search examples, pi.

Here you have the code to get started and basically ~~here is that, ~~get ads and then using these examples and then you can query on different jobs. We'll use [00:02:00] this first to explore the data, and then we will turn this into, dLT later on that we can connect it ~~to ~~to load the data into Snowflake.

Let's move on to Visual Studio Code now.

**Kokchun Giang-2:** Here I am in visual Studio Code and ~~we ~~let's start off with ~~UV VNV. We have ~~a virtual environment ~~source, ven ~~being activated. ~~Activated, ~~and then ~~UV ~~PIP install. We need to have DLT ~~Snow ~~Snowflake. Okay. And then we need to have IPI kernel for PYN notebook. ~~And I. Think this should be maybe Yeah.~~

Requests we need to get the data from the API. Okay, this is done and now we will create~~ actually I need ~~terminal. I will create, make the DLT code where I'll place the DLT code. I will have~~ worksheet ~~worksheets, snowflake. And this is it. I will also do touch gi, ignore, [00:03:00] inside of gi, ignore, I will make sure to remove dlt.

Because we'll have our secrets over there and we will also get rid of~~ let's see. And ~~DLT set up user ~~I will also get rid of. ~~Okay, now it's time to go into CD ~~worksheets ~~worksheet Snowflake. And inside here I will create a few SKL files. I'll create setup setup database sql.

Set up ~~users, ~~user s, kl, and then I will have check loaded data sql. ~~Set up and then we'll have ~~set up roles. Sgl. If you have watched the video before, it's a kind of similar setup. ~~You need to set, ~~you need to create the database inside of Snowflake. You need to create the schema. You need to create the roles, you need to [00:04:00] create the user for DLT.

And then ~~you give ~~you give the grants to the roles for different things. For example, ~~the work, the what's called. We need to give ~~the warehouse, the database and the different select and different ~~crowd ~~operations to different roles. ~~And then we also need to ~~and why I do Git ~~nor ~~in setup user is because I want to place ~~in ~~my password there and we shouldn't push the password up to GitHub if you're working in GitHub repository.

This is not a GitHub repository, but I put it in gi Ignore here to make sure that when you're watching this and following along, you don't push up your secrets there. And also same with ~~dot ~~DLT, because there we'll have our secrets ~~do two. Now I will see the up. ~~Now I am back into ~~route ~~and I will do C-D-D-L-T code and inside of DLT code, let's touch a few things.

Touch I want to have a ~~P Ed, IB. ~~This is the Jupyter Notebook that we will use for exploring the API. [00:05:00] We'll have load job ads do pi that ~~this ~~is the pipeline. You can see here that they exist now, and we will also do. ~~Make Dear ~~DLT and in not DTL r ~~dash rf dtl. Let's see make dear dot ~~d lt and then we'll do touch DLT slash secrets ~~two.~~

Okay. Let's start here. Let's go back to our browser and. Let's get back here and I will go into this copy, this part here. This is the DLTs connection to Snowflake. Paste it in here. I will call this job ads, this database. And the username is extract loader. You need to create this user if you haven't done that.

But I'll show you how to do that. And then this password, I will pause my video and I will set this up. And also ~~the host ~~the host you find ~~in in ~~in your account [00:06:00] details. Inside of Snow site, you click on your account and you click on account details. You will find your account identifier.

That is what you need. The account identifier you. Paste it in is basically a few capital letters and then dash a few capital letters. Okay. And then the warehouse, which is Dev warehouse, this is the one I created. But if you don't have that, you should either create that or you can choose another warehouse that you have.

And the role that we have is job ads, DLT role. This role doesn't exist yet either. This will create in our setup roles and the database will create, that's called job ads ~~There. ~~And also the user will create this extract loader. Now I'll post the video and I will set my own password and my own host.

**Kokchun Giang-3:** We'll start with setting up the user, and then for that I'll use the user role user admin. [00:07:00] Note that I am logged into my SNO sq l account here in the Snowflake extension. I'll run this one. I'm using user admin, and then it's great. Actually I can copy this one from the lecture notes.

Here, create user, if not exists, extract loader. Note that this one I place in the setup user and this setup user is get ignored. And then I will pause my video to to set up the my password and I will use the same password as I have in. Secrets two. This is very important. Otherwise, it won't work when you're loading the data later on in DLT.

**Kokchun Giang-5:** Remember to run the create user after you have put in your password. I have already done that. That is finished. Now it's time to set up the database. And to do that, I will use the role. Let's see, use the role si admin run this one. And then I will do create database, if [00:08:00] not exists.

Job ads. I'll create this database. And then I will do create schema, if not exists. Job ads do staging. Okay, now we have created that one and it's time to~~ do ~~set up the roles. I'll go into, set up the roles and change language to snow, like SQL. Now we'll create this role. A use role user admin.

I'll run this one. Create role, if not exists. And what do I want to create? I want to create job ads DLT role. I create this role and then it's time to you, you, I'll change the role to security admin. This is the one that will handle the grant, right? Grant the role. Grant [00:09:00] role job adds, DLT roll to user extract loader.

And also I'll grant this to myself. Job adds DLT role to user tion. Okay. I grant it to myself. And now it's time for us to grant and I won't I'll just copy this one as I have done this all before. I copy all this and I will explain it to you. First, we grant usage on warehouse, dev warehouse to roll job as DLT.

It needs to have the warehouse in order to do the compute, right? Grant, and then we need to grant usage on database job ads that it can access this, view, this database. And then we need to grant usage on the staging schema. And then we grant create table on schema staging. In inside of staging it can create tables which is needed [00:10:00] for DLT.

And then afterwards I will do. I will do crowd operations it can it, I will give it all of them. Select, insert, update, delete on all tables in the schema, and then Grant, select insert update, delete on future tables in schema. In in tables that we'll create later on, it will.

Still get it, it will still be able to do this within the staging schema, not in other schemas. We don't want to give it more privileges than it should have. Okay. And with this I want to run all of them. Click on this one. This will execute ~~from ~~from beginning to end. Okay. And now it's time to ~~show ~~check our grants.

I will copy that as well. Show grants on schema, job ads. The staging, you can see create table usage and ownership on the schema. On [00:11:00] the schema you have these things and then. ~~Future grants. ~~In schema, you can see we have the lead insert, select, and update. The crowd operations on all the tables within this schema is give this schema.

Staging is given to this granted to this job as DLT role, and then show grants to this role. And what you can see is that it has usage, create table usage on these different database schemas and warehouses .

**Kokchun Giang-6:** And then let's do show grants to User Extract Loader. And you can see that the Extract Loader has gotten this job ads DLT role and also you can do show grants to the user that you used yourself. I used Tion as the, it's my user. You can check that. You also have this role that you can ~~see the ~~see and work with the.

Tables that DLT will create for us later on. [00:12:00] Okay, this is setting up the role ~~and ~~and now basically we have set up a database. We have set up the roles, we have set up the user. And what we need to do now is to explore the data basically. In order to explore the data we have here I would call this, eDA on job tech API, because ~~what? ~~This is what it's called. And basically what we'll do first is that we'll go into the browser and inside here this is the code that we'll have. I'll copy this code basically. I will copy this code. I'll go back here, paste it in, change this to Python. Let's see, Python environments.

DLT. Extract. Load, yes. Okay, I will remove that one and just run this and see what we get. And then I'll go through it a little [00:13:00] bit. Okay. Run it. And we can see that we get a lot of teachers here. This is in Swedish, ladder it is S blah blah, blah, because we have queried, ladder sala. What it does is that we do.

Example, search loop through hits. Query it goes to here, into this one. And here we have search parents, the query, and then we have the limit, ~~how many ~~how many how much limit we want. And then the JSON response is basically this underscore get ads function. You can take a look into this, get ads.

What it does is that we have. The response here is the request get, and you have to send in this headers to get the back, the chason, and you put in the URL for search and the different paras that you have. This is the paras that you send in and you send in the search paras. And then you do response that this is to check for errors, and then you have JSON loads.

Basically you get this back [00:14:00] into JSON data. This is basically how you get the data from this API and the URL for search. You can see it up here is this URL here. It's basically this URL and then slash such. Okay. And with this one you can also do this example such loop through hits, and it will go through the hits and you will get to print out the headlines.

Hit employer and name. Now we have 11 hits, but let's move on a little bit. We can test out a few functions. Example. Search return number of hits we can do. Setting our query. And what you can see, we get number hits is 11. What else can we do? We can go into, we can take a look into the search risk.

We can go and copy these ones [00:15:00] here.

This is the search pars and this is the JSON response. And you can take a look into the JS response basically. And this JSON response, you can see that it's a dictionary, you can take a look into that. One type is a dick, and if it's a dick, you have the keys, right? Keys. And then the keys, you can see there's total positions query time, result time stats, free text con concepts, and hits.

Okay, then we can take a look into different things. We could do Jason response dot get. Hits. And then you see we get this is similar. You get a list of the number of hits. This is a list. You can do take a list. You can not take a list but length of this one. And you can see it's 11.

It's same as the one that we had [00:16:00] before where we have the number of hits here. Okay. What else can we do? We can do. We copied this one. This gets all the hits and we can take a look into one hit. The first one or the second one, if it's counting with starting with zero. You can see here we have one, one school here.

~~Y~~

~~This apply want teacher. And ~~what we can do here is since this is a dictionary, you can see that with this one calibrates and you can see key and value pairs. Then what you can do is dot keys to check out what's inside here. Yes, there's a lot of different things. Inside of hits, we have keys, right?

And then we can take a look a little bit further going in here. We can take a look into the headline. For example, headline, and we can see Jensen Naum. Okay. And we can [00:17:00] continue here. ~~We can ~~we can take a look into we go up here and copy this part here.

To get the, to search for something else and then you can see the query. Let's change the query a little bit. In, we can type it out here directly. I'll search for data engineer 'cause this is our field. And for data engineer, we can do get ads on that one. And also you can see, we can place this directly into this one here, we don't need to divide it if we don't want to.

And then we can take a look at the JSON response. Okay. A lot of jobs. 78 I guess. And 99 positions. Okay, great. Let's see. Dot keys. And it's same as before. What you can do now is that for example, we can take a look into Jason response of hits and ~~we, ~~it's a list we can take out [00:18:00] minus one to get the last one, and we can take out the headline.

And we can see, okay. Senior software engineer data and video analytics learned. Okay, great. And we can take a look into, I'll copy this one and take a look into their employer because that was one that we saw before. That, that is one of the keys. ~~And ~~and inside of the employer, we see here phone number.

We see email, URL organization number name is this Axis communications. And what we can do is that we can pick out the names if we want, we pick out the name. Okay, great. Now you can see that okay. It's a senior software engineer for this company. Okay, this concludes the EDA that we need ~~and late ~~and now we will go into the actual DLT code where we'll use this that we [00:19:00] have learned from the EDA to actually load the data from this API into snowflake 

**Kokchun Giang-7:** let's move on to low job ads. The DLT. We go in here and we import DLT. We import requests. We need the requests, right? We import JSON and we take from lib import path and then import os. These are just a few imports that we need. And then now we will go back to API key, our A-P-E-D-A.

And here ~~we have ~~we have these get ads. This is what I want. I will copy this one. Actually, I'll copy all these, actually, yeah, I'll copy this. Go back in here, paste it here. First we have this get ads. And here I [00:20:00] will put in URL for search. That you can get different ads depending on what you want to search for.

Okay. And then ~~we will do, ~~we will create our resource, at DLT resource. This is a way of grouping several data yielding. Data where we get data from. Different resources are the lowest unit, and they can be combined into a. DLT source. Which we will take from the source or the resource to the target.

And the target is the destination, which is snowflake in this case. We take DLT resource and we can choose the right disposition here, and I'll choose replace as the simplest one. But there's also append and merge for doing ~~diff ~~other types of rights. Depending on what you need, you [00:21:00] might be able to be interested in using the other ones.

Deaf job ads, resource. And here we'll take in paras and we'll take these two, I'll move them down in here, I move them with alt down button, or option down button if you're in the Mac. Here we have URL and URL for search. And now I will take four add in, underscore get, add. We take this function, we put in URL for such, and put in the paras the different parameters that we have.

The parents will be sent into here and the pars will be sent into this function. Get ads, and we'll use the pars here. On for our GET request. Okay? After we do [00:22:00] that, we remember that when we get the ads, we need to go into hits in order to get our jobs right. I will take dot, get hits.

It's a dictionary. We do not get hits. We get the hits and then we yield the ad. Remember that when we go into hits, we get a list and then we yield the ads. We send out one, we send one, we return one ad at the time. One item at the time. That is what yield does. It's yield instead of return, which will return everything.

This is a generator. I won't go in exactly what that means, but this is what you have here. You're yielding each ad when you're calling this. And this is required by our pipeline later on. Let's create our pipeline. I will change it a little bit. Now. I will put it, put the pipeline before in, in [00:23:00] another video before we put it into our script directly in if done the name equals standard domain, but now I will create a function of it.

I'll call it run pipeline. Send in a query and table name that you can customize it a little bit. And then pipeline equals to DLT pipeline. And here we have pipeline name equals to ~~job search.~~

~~And we have let's see. We have pipeline name equals ~~job search and we have destination equals to snowflake. Remember that this destination, it will look into our secrets to two and inside there we had specified snowflake. Destination and the configuration for that. Okay, after this part, ~~we want to go into let's see.~~

~~We have, we should have three. ~~We should have dataset name also, which is the schema. Staging schema or staging layer. This is where our data will land. [00:24:00] And then after we have this~~ we will run, ~~we will you take the pars equals to it. Remember that the pars is defined like this. We have q for our query.

What, whatever we type in here for query. For example, we want to search for data engineer and then we put in data engineer here in this query and we have a limit here. I'll make it default to 100. And I won't go through how to do pagination here, but basically you can do pagination and work with limit Together with offset in order to get more data.

I have that in the repo, if you are interested in you can take a look into that, but I won't go through it here how that works. I also have another video where I do, basically the same thing where I have the the pagination implemented. However, instead of snowflake, we used the, we used ddb.

[00:25:00] You can take a look into that video. I will link that in the description. Take a look into that if you are interested in how to do pagination. Okay. But I want do it here. We have the paras. We'll do load info equals to pipeline. Not run. We run the pipeline. And we put in, when we run the pipeline, we need to put in the resource or we need to put in the source.

We put in job ads, resources. You can see data is the first one. Job ads, resource, this is our data we place in pars equals to pars. Look at this job ads resource. It requires this paras. We're sending these paras with query and limit, it gets these paras and then it will send this into this, get ads, and we will return back the hits based on the query, basically like that.

Okay? [00:26:00] It means that we had this loan info. Which runs the pipeline and then we print this load info and that is it. And now we can do if done, the name equals to done domain, and I will do, I need to do working directory equals to path then the file.

Parent. Working directory, this is the Dunder Path parent. We're into this file, parent, we go into DLT code, but here. This working director, you'll be able to see dlt. That is because when we click on the run button, it it runs based on where you are in the terminal. If you are in this route up here, then you won't find your secrets or you won't find your [00:27:00] dlt.

That's why we have to create this working director and then we need to do this OS change deal. ~~Working directory. Okay. ~~And now I will ~~change, ~~do my query. I'll call data engineer and table name. I will call this data field job ads. This is the table name that I will send into my run pipeline. I send in query is data engineer, let's say query.

Query comma table name, and then we need to use the table name here. And what we do is inside our pipeline we can do table name. Equals to table name and basically is that we, okay, we have dataset name. This is the schema, the staging. And inside this, we'll create this table with this table name here, that this data field job ads.

We'll get this table basically, [00:28:00] save this one and I'll run this one,

and then it takes a little bit of time.

Okay, after it has run, we have pipeline job, search load, step completed one load packages where loaded destination, snowflake, and in dataset staging [00:29:00] job as location load package. Okay, we can take a look into our. Loaded data now. We'll go into our check loaded data and I'll change the language to Snow Snowflake SL and we will use the ~~roll.~~

Let's see, use role. Job adds, DLT role and run this one. And then we'll have use database

job ads. Run this one and I will show schemas. You can see which schemas we have. Yes, we have information schema, public and staging. Okay, we can show [00:30:00] tables in schema staging. Then we have a lot of tables here, and you can see we have. Okay. We have a lot of tables to see data, field job ads must have education level, et cetera.

And we have here data field job ads. There's a lot of data that has been loaded, oh, sorry. ~~It's, ~~what you can do is that you can select on all the ~~diff ~~different ones to see what data has been loaded there by DLT and see which data that is relevant for you, which you can work on later on. This is one reason why we placing everything, we dumped the data into staging and then.

In the staging layer, you pick out the data that you want and you go into the next layer, which is the refined layer and where you can do your [00:31:00] transformations. Here I want to just take a look into this staging.data field. Job ads, it's called. I will describe table, take a look into this one, and you can see, oh this is the ones that we get from the hits.

We get relevance Id label, webpage, UL, headline, et cetera. Description, text. There's a lot of a lot of different columns here. Then we can use. Let's see. We can do select headline employer let's see what we can pick. Employer here. Let's see. Employ, let me see.

Employment type here. Were employment type. Yeah. There it's not [00:32:00] alphabetically, but yeah, we can. Just employer workplace. Maybe there should be one if they haven't changed it from staging.data. Field job ads. Run this one. Yes, it works. Here we have a employer workplace. What you can do more is basically, for example description, text.

~~We could take. ~~Yes. And you can see here is the description formulate h and m job description. Okay. Data engineer, air tele two. Yeah, it's a little bit longer description. Good. We can do yeah, basically here ~~you can ~~you can do a lot of things. ~~You want to select the. ~~We can perhaps pick out one one employer, for example.

Okay. A TG. Do we have more things there? ~~Select ~~select star from staging, do [00:33:00] data field job ads~~ group by or where we can take, ~~where employer. Workplace equals a TG, maybe? Yes. And then we only had one, right? Yeah. One data engineer. Okay. now you can work with this as you want. I will save this one and I will show you that the data actually have been loaded into Snowflake.

**Kokchun Giang-8:** Inside of Snowflake or Snow site, we can see that we have this data field job ads under. This is the job ads database. And here we have the staging and we have a lot of tables. ~~We have this data field job ads, and ~~you can see the data preview here. It'll take a little bit time for it to load.

I can see this because I've assigned the role ~~to ~~to myself. ~~I can see that. ~~And here you can ~~see ~~see the different [00:34:00] things that we loaded. You can see here the one that we got before data engineer at reflex, this description, text, et cetera. ~~Okay. And ~~and you can see there's a lot of columns here.

66 columns. You can take a look into that one. And what DLT does is that you can see how ~~the ~~it has nest. Basically you have nested data, right? ~~You have a lot of ~~you have a json object with a lot of nested. ~~And and the nested ones become two underscores and. ~~And in DLT it's possible to load based on the level that you want.

But when it goes even deeper level, you can see if you drag it out here, that it's continuing another nest. Here, data field, job ads, and then underscore, underscore application contacts. Actually, this one could be very ~~u ~~useful for you if you want to. You could join the tables later on.

Here you can get the different emails for different people that if you want to apply for these jobs, you can take their emails and their phone numbers~~ and ~~and you can use it. And also here [00:35:00] you have for example, here's another one that is quite nested. You have here education.

And you can see the data preview ~~and, ~~but there's not much data there, this is also something you could take a look at and go through each of them. ~~Okay,~~

**Kokchun Giang-9:** in this video we have used DLT to load the data from the job tech API, and we have extracted and loaded it from the API into snowflake. And then we ~~go want, ~~went into working with snow SK l to actually ~~test, ~~test it out and saw that, yeah, we got data we could ~~do, ~~query on the data, and then we went into snow site and checked that the data was there.

But before this, ~~we were ~~we needed to set up the database, the roles in we use snows well, but you could ~~do, use it, ~~do it ~~in ~~in worksheets, in Snowflake in snow site. That will also work. But yes, we have gone through the el part of the ELT. This is the start of your. Pipeline where you ingested data from an [00:36:00] API.

This is super cool. Thank you for watching this video and hope that you have started to get interest in Modern Data Stack and continue to learn more. See you in the next video. Bye.

