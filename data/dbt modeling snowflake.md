# dbt modeling snowflake

**Kokchun Giang:** [00:00:00] Hello and welcome to this video on DBT modeling. This video is based on some previous videos, ~~it's based on, ~~one is the DLT video where ~~we have loaded data into, ~~we have loaded job tech data, job ads, data into Snowflake. There we also set ~~up. ~~Up the users. We set up the user for DLT.

We set up different roles for DLT that it can access the staging layer of our data warehouse. This is based on, first this one. And then it's also based on another video where we've talked about setting up DBT. And when we set up DBT, we also did some worksheets in Snowflake where ~~we did ~~we created the transformer user, we created the job ads DBT role.

And then we also made sure that we could connect to Snowflake through our [00:01:00] profiles YAML and our DBT project that we can do DBT debug that it works towards Snowflake. Based on this, we will continue on after we have done this setup also in the setup, ~~we did ~~we set up DBT Power User in Vision Studio Code.

Now when you have all this set up, it's time to continue. To create our DBT models then we will have based on our dimensional model, which we also have covered in another lecture. However, I'll show you that image, that we'll follow that one. And I won't do all of the models.

Instead, I will just do ~~one , ~~one fact one. Dimension one Mars. ~~And ~~and then, because I don't want to ~~take out ~~take away all the fun for you. I'll just show you all the core principles and then you can go on and do all the fun yourself. [00:02:00] Let's move on to Visual Studio Code now.

**Kokchun Giang-1:** here I am in Visual Studio Code. What I have is I have a virtual environment. I have activated it, and I have installed, I've done UVP install DBT dash core, DBT dash Snowflake, and DLT since we'll be using DLT to load some new data as well. That is what I have, and then we have here, I will go back a little bit.

Worksheet Snowflake, what have we done here? We have in the previous video where we've gone into DBT setup we needed to, we also did some setup in our snowflake. What we did is set up the user. Here we created this user and then we set up the warehouse LA we set up we granted the user these different roles.

~~Job ads, DBT role, ~~job ads, DBT role. And then we did set up the marks layer. We did set up [00:03:00] the warehouse layer as well, since the staging layer we have already set up before. What we do here is that we just set up these. To layers. Now we have three layers. We have our we start with the staging layer, and then we have the.

Warehouse layer. And then finally we have the marts layer. These layers are basically just schemas in in snowflake. Okay. We have done this setup and then we also have this job ads dimension model. This model here, you can see it's a star schema and this is a quite good model in order to to make it understandable, to get the mental clarity of the of the models that you want to create that you can get mental clarity of the data.

Basically we have the fact in the middle, and then we have dimension occupation. We have dimension job [00:04:00] details. We have dimension employer and we have auxiliary attributes as dimension. These are dimensions that will enrich the fact that ~~you can ~~you can see the fact as an association table or some kind of bridge table where you have.

A lot of foreign keys to the primary keys here. They're connected through these. And then there's also a few facts here that are the vacancies integer and relevance in float, and also the application date in the fact job ads. For example, you want to know more about ~~the job detail, ~~the job details about this.

Particular job, then you just join this one and you get all these columns as well. That is the idea. And then you can choose what to serve for different Mars in the Mars layer later on. Okay. And we also want to go through in the DBT code. What we have here is the DBT project yaml. Here we [00:05:00] have set the, this is the name of the project called DBT Code.

And here we have the profile is the DBT Snowflake. This must match the profile in your profiles yaml. You reach the profiles YAML by doing this open tilda slash d. BT slash profiles yaml. If you click okay there, I won't do that because if I do that, you will see all my credentials and that's very bad.

You should open this one and you should change your credentials and ~~you should point, ~~you should have one that is called DBT, snowflake ~~and I can also. ~~Note that if you are, for example, following this course, ~~and you can do one that is for your specific DB t. ~~You could call it DBT snowflake course if you want or whatever, but it must match the one in your DBT in your profiles ~~that yamo.~~

And then you could have one that is called DBT Snowflake, the underscore project, for example. And then you need to match that in your YAML as [00:06:00] well. You kinda, several. Profiles in your profiles yamo. And inside of profiles you have your setup. ~~You choose, you have which account identifier you should have.~~

You have the password to your account identifier for your Snowflake account. And you have the user, for example, the transformer user that we have created. Four with a password for that. You have created this user ~~in ~~in the previous video, in the DBT setup. ~~Then you know that you, I will just, ~~I just wanted to point this out that this needs to match.

~~Okay. ~~And then scrolling down here, you can see that right now we have. The DBT code, we have the staging and we have the refined. Here we have staging for it is called the is the staging schema, and refined we have the. Warehouse schema here. If you have models here, you have staging part and you have a refined part.

Here ~~the refined parts is ~~you have the staging, basically you have the. Inside here you have ~~your where it's the, ~~where it's [00:07:00] connected to the warehouse schema in your snowflake or in your data warehouse. And the staging folder all the models here. They will connect to the staging schema in your snowflake account.

Okay, ~~I, ~~I hope that this was not very hard to understand. If you think this is hard to understand, go back and make sure that you follow all the other steps in the previous videos fast. I will link all the previous videos in the description you can take a look into that. And let's go in here.

Let's show the DLT as well. Here, the DLT you need to put in your secrets for D lt. There's ~~do ~~DLT inside there. There's secrets ~~Doum, ~~I won't show that. But in our, ~~in the re ~~repository, go into the D-L-T-A-P-I part, then there's how you set up the secrets and also in. If you are not sure about how to set up the [00:08:00] DBT the profiles at YAML in DBT, you go into the set up DBT folder in our repo.

~~I, I will, ~~I have everything in the repo. We have divided into several lectures. Here in the load job ads, this is a little bit different from what we have done ~~in in ~~before, when we loaded, extracted and load the job tech API before we did the, with just a query data engineer. And now the difference is that we want to have technical jobs.

What we have done is here. You have a particular occupation field with ~~this num ~~this ID here, which you can search up for in the documentation, but based on these, you will get the~~ when you run this, you'll get the technical job ads. You get another table called ~~technical job ads. And this is the first one that I will start with in this lecture here.

~~Otherwise. ~~Everything else is the same. It works. By doing this, ~~you do the, ~~you do a get request basically, and you have a. Resource here where you yield each [00:09:00] ad and then you have this in a pipeline and ~~you ~~you extract and load the data from the API, you load it into the staging layer or the staging schema inside of Snowflake.

And when you do that, you do this pipeline run job ads resource, and you have this table name. ~~Which is defined here. ~~That's technical field job ads. Before we had something called data field job ads, you should know that. And this always changed. There is just that you are in this folder that you can see this DDLT.

That is it, I will run this one first. Start running it. You can just click the run button if you have activated your virtual environment down here. It takes a little bit of time to run this one ~~and.~~

~~It, ~~it's loading the data and here it says yes, it's done. Load step, took much time, load, package is loaded and contains no failed jobs. And when it has done this, you should go into Snowflake to take a look into your data. We'll take a look [00:10:00] now.

**Kokchun Giang-2:** inside of Snowflake we have here we start with data field job ads. This is the previous that we have loaded, but now we have loaded another one called Technical field job ads. This is what we just loaded and you can take a look into DLT loads, for example. And data preview, you will be able to see that we have just loaded here, 0 9 0 7.

This is the most recent one. This is UTC time. It's two hours behind our clock. Now we have technical job field ads. This is the data that we have loaded. You can see data preview here, and you can see here is the data. And you can query this if you want. Good. Now we know that this data has been loaded, let's move back to vicious Data code and DBT to do some modeling.

**Kokchun Giang-3:** back in Vicious Studio Code, let's start with something very simple. We need to go into [00:11:00] CD DBT code. Now I'm inside of DBT code, which means that I can see my DBT project at yaml. If I do ls, I can see DBT Project yaml, and now we will just do DBT debug to see that everything is connected correctly.

Yes, and you can see that it works. If it's not connected correctly, make sure that you connect it correctly. ~~And I'll show you the read. I'll just show you the in my, ~~I'll show you in my GitHub~~ how ~~how the connection should be.

**Kokchun Giang-4:** inside of setup, DBT in my GitHub repository, you can see in the read me here that ~~it should, ~~it says in the profiles, yaml which you find~~ in ~~in this directory. Your home directory, dbt ~~slash ~~slash profiles yaml you have here, this is what you should put in. ~~Db, tco, snowflake. ~~This should be connected to the same as [00:12:00] the ones in our.

DBT projects yaml. Then you have here target dev. You have outputs dev, and then you have Snowflake. And this is the user transformer. This is the password that you have used when you created your transformer user. This is very important. They must match. And then here the account identifi. You find it ~~in your your pro, in your ~~inside of snow site.

You go into your own account and then you find the account details. And inside there's account identifier. You fill this in here. And then when this is done, let's move back to Vicious Studio Code. ~~And we, ~~when you do DBT debug and it seems ~~to, ~~everything has checked, then it's fine. You're good to go.

Let's start. This is a copy from the 10 where we did the setup, but I will not need refine or staging. I'll remove this. Actually I'll remove this one as well to make it a clean start. In the models. We'll create a few folders. I'll go into CD [00:13:00] models and here ~~I'll make, there, ~~I'll create dim fact marked and source.

These are four folders that I will create here. And the reason for this is that this is how I want to structure my DBT models. I want to place all the dimensions. In my dim, I want to place all the facts in my fact and all the marts inside of Marts and Source is the ones that will connect to the staging.

Here, we'll have something else here as well. I'll show you later. Soon, however, let's take a look into our DBT project at yaml. This is very important that you have here is DBT code. This is the same as the project name. Your project name here is DBT code. ~~It's, ~~it should be the same here.

And staging in schema staging, yes. Materialized as here I want this, I want it to be [00:14:00] called I don't have a staging here, right? I want to change it to source.

**Kokchun Giang-5:** here ~~we, ~~we changed it to source, and actually I can delete all this. We don't need that. Source, and then we'll do plus materialized. I want it to be materialized as ephemeral. And then I want to have plus schema staging. And this plus is just to denote that this is a resource config and this is somewhat it's backward compatible to older versions of DBT.

But I like to use the plus in order to denote that this is the resource config and it means that there's a lot of configurations we can do here for the resources. That is how DBT should build or run these models and where it should be. Stored how it should be materialized. Ephemeral means that it's a temporary table.

We materialize this DBT the models in the source as ephemeral that you can just [00:15:00] use them there, but it's not stored in Snowflake at all. And that is the idea. And the reason for this is that you have the source. For the reason that you can actually, you can do testing on your source data, and that is really useful.

You can test to make sure that this, the data from the source is is validated before go moving on that it's, make sure that your pipeline doesn't break that you know you should test it in different places. That is the idea with the source as well. ~~Here we have, I want to have another one plus materialize.~~

And here I want to make it inside and I want to materialized as table in for more general. This means all that are not specifically decided how it should materialize will be materialized as tables. Okay. And then we have dim the dimensions should have schema. And I would choose the warehouse schema.

This one we have set up before and then we have fact, [00:16:00] it should be in schema. Let's see. Warehouse as well. We note that staging is the landing zone where all the data will land, and then afterwards we do the dimensions and we do. Facts in the warehouse layer where it's the more, the refined layer, usually you can call it warehouse, you can call it refined layer.

Now we have picked the name warehouse, we're stuck with that one, but that's totally fine. And then we have the Mars layer, and here we have plus schema. It's inside of Marts. ~~These. ~~Very importantly is that these must match the names here. You have dm, FCT, mark, and SRC, you have SRC, DM fact mark, because here our DBT models will reside.

We will have our DBT models here, and then you can see the actual schema here is the schemas inside of snowflakes that will be changed. And note that this is just, I'm just talking about Snowflake right now, but it [00:17:00] could. ~~That's ~~easily be something else. For example, BigQuery, it could be Duck db, it could be something else.

What you need to change is your DBT, the profiles that yaml that you, you choose here that you're pointing your profile to something else. Yes. And that is it, I think DBT package. Yeah, that is totally fine. And. Yes, I'm happy with this one DBTD bag to see that it still works. Yes, it works.

Clear this one. And now let's go into our source and we'll start with creating something called.

This is our starting point of our journey for DBT modeling. Sources you define here in this YAML file dash name, job ads. And then here, let me do spaces here. [00:18:00] Schema is staging. Here we'll pick our data and I'll explain why we do these tables. And inside of tables we have Dash name staging ads, and we have Identifier is Technical Field Job Ads.

Okay. Remember the name of the table is Technical Field Job Ads. This is the name of the table. Okay. And then we have. We give the name here, stage ads, and then you have the schema is inside of staging schema. This is where we find the data. It is this schema and this table name. And then we have named it Stage ads, and we have named this job ads.

The reason for this is that. For example, it could be changed in your in your staging data in your staging layer. The [00:19:00] this name could be changed. And the reason for we doing this in the sources that YAML is that when it changes, you just need to change in one place. You just need to change in this.

File here, this YAML file and all of your other models, ~~they, ~~they don't need to change at all. They can be the same. Otherwise, if the name changes here ~~in, ~~in for example, in Snowflake, someone has changed the name of this table. And then you need to change all your model names where you refer to the data.

And that is quite tricky. You want to change it in one place. Good. Now you have. And we will create our SRC job ads, sgl, and we'll create, yeah, we can start with this one. This is a DBT. You can change language mode, and you can choose here d. SQL, you have this SGL, and you have configured [00:20:00] it to be DBT D-B-T-S-Q-L.

Start with note that this is an extract of the model. It's not the whole model because ~~I want, ~~I don't want to ~~give, ~~take away all the fun for you.

**Kokchun Giang-7:** Okay, so let's start creating this model. But first you need to change to right language mode. So shift control P or shift command P. If you're on Mac, change language mode and you choose here, ~~Jin, ~~SQL. So you click this one, Jin sq l, and when we choose this one, we will have the power of power user. So this is really cool.

So with stage job ads. Select star from, and now let's see this magic ref and then enter. Cool. So now we'll see here Ref. And you choose the model name that you want. Or actually, I don't want ref, this is not the ref I want. I want to have a source, actually. Source enter. [00:21:00] Ooh, this is so cool. So we pick the source and the table name.

Okay, so let's take a look into Saucers. Yamo Source is the job Ads. So this is the name. And then we have here name of the table, stage ads. Okay, so source job underscore ads. That's it. And here we have stage ads. So remember what I said before. So here it means that we have so whenever, if we change something in the Snowflake table name we just need to change in the sources because everything else is, that is connected to the sources is is done through this one.

So you have this CTE here. And you have the source job ads and stage ads, but these stay the same. Job ads here, the name here and this name stage ads stay the same. And the only thing you need to change is this [00:22:00] here and the, or this. So that is very cool. Okay, so we have this one first and then select and here we can just select a few of them.

For example we can start with select star from stage job ads. And let's see, stage job ads because it's this CTE that we're working with. And let's start with this one. So control, enter. Yes, and the DBT Power user is run. ~~So this, so ~~note that the power user is not needed. You can do this directly with DBT Run, and you can see the results in in Snowflake when, unless it's ephemeral that it is in this way, in this case.

However the thing is that it's quite good to have DBT power users so you can. See the query directly. However, this could get stuck sometimes. But it's let's see. Yeah, maybe it's gotten stuck.

**Kokchun Giang-8:** So it seems to be this error [00:23:00] here. DBT found one packages specified in packages yaml, but only zero packages installed in DBT packages. ~~So run DBT depths. ~~So let's do that. DBT depths actually I'm in models. Can I do this? I need to go backwards. CD DBT code DBT depths.

Updates available for this one. Update your versions in packages, yaml, then run DPT depths

**Kokchun Giang-9:** so basically it works now and what I did was okay, I just changed I went into my DBT, the packages that YAML and I changed it to 1.3 0.0 because that was what the error stated. Or the message stated that there was a newer version of this one. ~~Okay, fine. ~~Then I did DBT depths again and updated it.

But still, DBT power usage doesn't work. But then I just, restarted Visual Studio Code. So close it down and start it again. And now it works. [00:24:00] So run it and you can see, run it with command, enter or control enter if you are in Windows, and you can see here is the data. And the thing is that it uses Snowflake ~~in in ~~as the compute to do this so that you can.

So when you get the data here, you can find out which one that are interesting for you. And the things that I think ~~is ~~interesting is that if you go into job ads~~ dimensional ~~dimension model here. And then here we'll need a few things. So we will need, for example, the relevance, right?

We will need the vacancies here. We will need application deadline, and we will need some other things. And the thing is. I will pick this and create based on this fact job ads, I will create this source because later on ~~in, ~~in the fact job ads, it will use this source job ads. So then we will create here.

So we'll have [00:25:00] occupation

label, we'll have number of. Vacancies. So these are ~~actual tables here. Or ~~actual columns here. And we will have this as vacancies. ~~We will have, let's see, comma, ~~we'll have relevance. We will have. Application deadline. Also, one other thing that you do with source job ads is that you're already filtering out which type of columns that you want to have for your downstream models.

So here is upstream ~~is ~~very close to the source very close to the actual source of the data in the staging. ~~And ~~and afterwards you serve the downstream, which is~~ in the. ~~In the refined layer or the warehouse layer. ~~So here, this is what I want. ~~So remember this is an extract of the model.

It's not the full ~~model. So extract of the ~~model. And here you can see these are a few things that we get ~~it. ~~And then we ~~order this one ~~order by application [00:26:00] deadline.

And you can see ~~here's ~~the application deadline here. Yes. ~~Zero ~~0, 9, 10. Okay. ~~And ~~so let's move on to our next source. Create another one, SRC, occupation, sgl. ~~And now ~~we should change this one as well to ~~Ginger, sgl, L ~~Ginger, SGL. Good. Now let's create this one. And similarly we create this one. So with stage.

Job ads as select star from source and then enter. And here you pick your source. So job ads. And here you have stage ads and then select. And here you have different things from stage job ads. So this is the CTE ~~that this ~~name here. [00:27:00] And now we'll pick occupation. Occupation, ~~group. ~~Group concept id.

So now what this is, source occupation is basically, if you look into this here, is that we have this dimension for occupation. So ~~these are ~~we need to get all those that are relevant to occupation and place it into source occupation, and then we use this later on to create this dimensional model.

~~This di dimension. ~~Okay, so we have here concept ID s, occupation, group id, and we have here occupation. Let me just copy this occupation group. The occupation field. Concept ID as occupation. Field id, I'll copy this again. [00:28:00] Occupation label. It's ~~two ~~like this, occupation. So simplify it as this.

And then we have occupation. Group. Let's see. Occupation group. Occupation group underscore label. And in order for you to find this, you should take a look into the data and see it side by side, ~~what type of fee ~~what type of columns can we pick? So this is a detective work that you need to do. So ~~I'm ~~already done this, so that's why I know which one I need to pick out as occupation group.

And then finally we have occupation field label as [00:29:00] occupation Field. Okay, so I run this one

and you can see yes. Occupation, group id, occupation, field id, occupation. Occupation, group. Occupation field. So these are the five that we've picked, and we have renamed it a little bit here to make it simpler. So that is what you can do with the source models. You can do some simple renaming and then afterwards you have ~~this, ~~these models, which you can build your dimensions and your fact.

So let's move on to the fact. Here we have fact job ads dot sgl change the language mode to SKL, but J sgl, okay [00:30:00] with job ads s select. So you always start with A CTE in DBT models. So select star from. And now we don't take from the source, we take from reference and the reference, we choose the source name, basically the model name.

~~So re ~~so this is so beautiful ~~with ~~with DBT that. All the references, all the sources. It's based on the model names which if you change the model names, you need to change this as well. However, it doesn't have the the hard coded references. Instead the, when you compile the code into the target, you will have the hard coded references.

So it's okay, select star from. ~~This ~~this actual table name where you have to go into this database, this staging, this layer, and then ~~this ~~this table. It's fixed ~~in ~~in the target where you have this, the output that SKL file which [00:31:00] comes from. Comes from all the compilation of your DBT models, which you do when you do DBT Run.

I'll show you that later. Rev source job ads. So we take from this source job ads. So this is the upstream model that we pick from and we do select. And here we will do like this dbt u utils generate. We'll actually do like this.

So this is how you pick the some functions. And we pick dbt, uts.

Dot generate surrogate key. So this is a macro. So Jinja means that Jinja, SKL means that it's [00:32:00] SKL combined with macros. And macros are basically just functions. ~~It's functions that exist. ~~And this function generates surrogate key. It's in this DBTU TILs package. That's why we installed this package before, and what it does is that it generates a surrogate key.

**Kokchun Giang-10:** So it's here we want to create this surrogate key and it should be based on a column name. And we ~~go, ~~can go back to source job ads. And you can see this is occupation label. I will pick this one as the column name and we'll put it here. And this means that this one will be the surrogate key, and here what the surrogate key means.

~~You will see occupation. ~~Let's see as occupation id. So it's basically just ~~a ~~a key that we have created based on previous columns, which we have done some processing to. ~~So I'll show you. ~~[00:33:00] So let's see here from job ads. So this is the job ads. I'll run this one and you can see here.

So this is the surrogate key. It is a hashing function. ~~So if I remove this one, ~~let's see ~~what the, ~~what we'll see here. ~~So remove this one, ~~and you can see the raw function here. The raw function is MD five ~~cost esque. ~~Cast occupation label as text. Okay. So what this does is that we are casting the text from this column name ~~into ~~into MD five format.

So it's a hashing function. So we hash this column basically, ~~and you can. A combination of columns, ~~but importantly is that when you do hashing function, it means that one input will give one output. It'll always be unique in that way. ~~So that if you are doing this for other.~~

Tables as well or other dimensions, you need to do the same columns in order to have it linked ~~to the same to the right ~~to the right row, right? So this is quite long name, that's why I renamed it. Using an alias here [00:34:00] as occupation id. Occupation id, that fits very well. And then we'll continue here with vacancies.

I will pick that one and I'll pick relevance. I will pick application deadline as well. So these are the things that I will pick from my source. Job ads. So if I run this one, it'll look like this. And note, this is also an extract of the whole model, so it's not all of them. So you can see here we have occupation id, vacancies, relevance and application deadline.

Okay? So that is the fact. And we'll go further to the dimension here. And here I will have one dimension. So deem occupation. Occupation sq and here we need to change the language mode to SQL. I'll go into JSQL here. So same as before with source occupation as select star from a ref. [00:35:00] And let's see, ~~ref, oh, sorry, ~~ref going down.

This one, you should click for power user and ref model name. We go into ~~source, occupation. So ~~source, occupation. So this CTE. Here we can use the source occupation. So select let's see, select here. From source occupation. And what do we want to select? Same here. DBTU TILs dot generate surrogate key.

And here I want to use my occupation. Let me choose occupation. You can see source occupation you have here this occupation label. I will pick this one ~~as my. ~~As the one ~~for ~~for doing my key or my id, so as occupation id. So this is my primary key and ~~I will so ~~in my foreign key, I will connect to this one.[00:36:00] 

Okay. And then we'll pick occupation as well. ~~We will here. ~~This is quite special. If we pick occupation group and we'll pick occupation field. So if I run this one it'll come soon. Yes. And occupation group, the thing is that you can see there's here occupation group. ~~We in technical in technique, ~~and then ~~we have the same again. ~~We have the same again. But the thing is in the dimension, what you want to do is ~~that you want to you want to make it into, you want to ~~make it unique, and in order to make them unique, what you can do is ~~like this you ~~group by the occupation.

So you group by this occupation. ~~So this is the ones that you want to make unique. So you group by all of them. ~~Then this will be unique. And then when you group by, you need to have an aggregation function. And we do max here. So it'll just pick the top one. So max of this one as occupation [00:37:00] field, and as.

Occupation group. And if I run this one

now, it should be fewer rows. Exactly. And here you can see these are unique. Now or the unique thing is from the occupation here. So ~~you have ~~you have all the unique occupations, and then you take the top ~~from ~~from each of these ones. So that's why you use Max here. Okay. So now we have our dimension occupation, and it's time for us to go even further. We want to go into our marks.

So this is the downstream ~~of ~~of the. ~~That the, this this ~~warehouse layer, so the refine layer or the warehouse layer, and then we have the Mars layer. [00:38:00] And for the Mars layer we'll create a file called Mart Technical Jobs sgl. I note that you might have several more dimensions and according to this dimensional model here, we have more dimensions, but you should do that, not me.

So change language mode s scale, GSL. And here we do with now. Note that this is the next step. So we take fact job ads. As select star from ref. We'll take this from our fact. So fact job ads. So this must be same as this name, ~~comma ~~deem occupation as [00:39:00] select star from ref. Deem occupation,

occupation. And now with the select,

let's do select star first from fact, job ads as F,

and then you can join. What you want. So for example, here we can do left. Join

deem occupation O on F dot. Occupation D equals the O Occupation. Id so note that both had occupation id. So if you look [00:40:00] into deem we have occupation id it when it used this occupation here and fact job ads, it has occupation id, it used this occupation label. They are the same because ~~we used, ~~we had source job ads.

We have occupation label here.

And we had this source, occupation, it's occupation label, but I, maybe I shouldn't have changed the name to occupation here, but ~~that, ~~that I've already done that now it's fine. It's still ~~the same column. It's ~~the same column. Note that source occupation refers to this occupation label.

Source job ads refers to this occupation label. So it's the same column. I just mistakenly changed the name before but now when I've done that, I don't want to change it back. So in the Mars, you can see we go into F, occupation ID and o Occupation id. Why is [00:41:00] this possible? Because DM occupation will have changed it to the occupation Id.

Fact job ads, we have done it. Occupation id And since we are doing surrogate key for these, the same column, it will give the same IDs and we can join them. So we join this

And let this run this one and see what happens.

So here we get occupation id, we get vacancies. Relevance application deadline. So these are from the F. So fact job ads. And then we have occupation ID two. This is from the O and occupation, I occupation, et cetera, or everything that's related to occupation. That is very good. So we have joined this together and then we can pick the [00:42:00] things that we want.

F vacancies, F, relevance, O Occupation. O Occupation. Group

O. Occupation Field. F application deadline. So now I pick the things that I want here.

So these are the things that I have. Now, maybe I can take occupation first. ~~I,~~

~~yes. ~~And these things, the Mars are the things that are sent downstream. So it's sent downstream so that we can connect this with, for example, stream it. We can connect directly to Mars. We can connect, for example, type high to Mars. We can connect the other front end to Mars, maybe, power BI Tableau.[00:43:00] 

And the thing is, these different marks, you give access to them based on different departments, different people. So you serve data, but you don't serve all the data. You choose the data you want to serve to the specific person or the specific entities. So that is the idea with the Mars. And here we have.

If we want to mark technical jobs, we could also do like this where all that occupation field equals to this metech in. So we just take out these jobs, for example. Okay. Maybe all the jobs I picked already had this only had this metech ing. But if you are ingesting several more jobs, then then definitely you can filter this

**Kokchun Giang-11:** okay, now it's super fun. Let's. Try to [00:44:00] materialize this now. So before we do that, we go into this generate schema name. This is very important that you have this script here because this one will make sure that it will enter into the correct names ~~in ~~in Snowflake. So I won't explain exactly what this does though.

This is just the macro that I've taken from the documentation of DBT. So what you do now is to materialize you the DBT run. Remember, you are inside of DBT code and you do DBT run DBT not found, right? I haven't activated my virtual environment. I go out DBT modeling. Yes, this is true. So source van being inactivate cd DBT code DBT Run.

And it found five models. So 1, 2, 3, 4, 5, but it only ran three. Why is that? [00:45:00] Because we don't materialize the sources. They are ephemeral. So this is totally correct. So we have 1, 2, 3. So now let's move into Snowflake.

**Kokchun Giang-12:** So inside of Snowflake in job ads database, we have these four schemas. So we have the mar schemas, we have public schema, staging, and warehouse. Okay, take a look into staging first. And now this is not strange. These are just the data fields and technical field that we showed before, which we gotten from DLT, right?

So nothing else here. We haven't, because our resources are ephemeral, okay? They don't materialize into tables in Snowflake while the other ones becomes tables. So we take a look into warehouse. This is our next layer after staging. So here we have tables. And we have, okay, this updated [00:46:00] headline and updated headline micro, this is just from a previous test in the previous lecture we have this, however, this deem occupation and fact job ads.

You can see here, fact job ads, data preview, and you can see the data is here. Occupation id, vacancies, relevance, application, deadline, theme, occupation. Data preview. Yes, it's here. You can see occupation id. Occupation. Occupation group. Occupation field. Great. And then take a look into Mars MAR technical jobs.

So these names are the same names as those that we have picked for our SQ L models. Click on this one, data preview. And this is the data that we're serving. So this is the data that now we can choose who should get access to these smart [00:47:00] technical jobs. Maybe only the technical department will get this, or maybe the marketing department we choose, right?

And we have several marks that we can give out to different departments. So you can see here, occupation vacancies, relevance. Occupation group. Yes, it's here. Occupation field, et cetera. ~~So cool. ~~So cool. We get, we've gotten the data inside of Snowflake after we had done the DBT modeling, so this is so good.

**Kokchun Giang-13:** Wow. I will just say, wow, this was a very long video, but totally worth it. We have used DLT to extract and load the data from job tech API, into staging layer of snowflake, and then inside of the staging layer we have used DBT models. To do we have created several DBT models to do the modeling, and then when we have done the modeling, we do DBT Run and all the models that we want are materialized inside of [00:48:00] Snowflake.

And then we have the data in, divided into the layers that we want staging. Warehouse layer or the refine layer and the Mars layer where we have served our data. And this is a note that I didn't do the whole modeling, I did just very simple parts so that you can use this later on for your, consume it for your later applications.

For example, using streamlet , using Tai Pi, using react, et cetera. Or you can use whatever tool you want, you can use Power Bi Tableau, and later on ~~show ~~show the beautiful dashboards to your different stakeholders. I hope that this was a worth even if it was a long video, I hope that you stayed and fought with me and learned a lot.

So thank you and see you in the next video. Bye.

