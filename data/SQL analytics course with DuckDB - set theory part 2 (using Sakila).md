# SQL analytics course with DuckDB - set theory part 2 (using Sakila)

[00:00:00] Hello and welcome to this video, which is part two of the set operations in the coding. Hands on part two. Here we will instead, like in ~~the, before ~~the first video, we generated some data or we inserted some data and did some set operations. And that was like very simple data to work with.

However, now we'll use this Quila database, and this is more. ~~Like very big da, ~~not very big database, . It ~~has ~~has several tables and we will show how to use different set operations in the Quila database. Since this video, it's the first time we'll work with the Quila database.

I will show you ~~from ~~how you will ingest the data into Duct DB to work with, because this is originally my SQL database that I have created~~ however, ~~in Kaggle, which we will go to soon. You can download the database file, which has been ported to SQL light and for us we need to [00:01:00] be able to read that database.

~~Yes.~~

**Kokchun Giang-1:** here I am in the browser and you can search a Esq Light Quila sample database and you will get you'll come into here in Kaggle and then just download this database and you will get an archive zip. Unzip it and copy that folder into Visual Studio Code. I'll meet you in Visual Studio Code and I will show you this.

**Kokchun Giang-2:** here I am in Visual Studio Code, and you can see I have copied in the folder into a data folder. However, ~~this is ~~here we have a lot of SQL files to actually generate ~~this ~~this data in SQ light, but I won't need that, ~~I will just delete that one. ~~Read me txt. It's good to have if you want to read more about the Takeda database.

Basically it's a fixes database designed to represent a DVD rental store. That is what you need to know. And then we have the PNG here, right? This is the actual ski [00:02:00] mass. ~~The, this or ~~this is the relationship model the data model. You can see as a entity relationship diagram.

You can see the entities and. What type, these are the tables and what type of columns they have and how they're connected to each other. I won't go into detail how this work because this is more the scope of a data modeling course. Right now we are ~~in ~~working with ~~SKL ~~and Duct db. What I want to do now is to read this~~ ESQ ~~Light database into D db.

To do this, we do like this. I start a new folder. I call it SQL, and inside this folder, I will create a file called Load ~~ula sql, and ~~to load ~~ula ~~in~~ induc ~~db. What you need to do is. Install SQ light. This is an extension. We need to install it, and then we need to load SQ light [00:03:00] extension.

? And after you've done that, you do call this sq light underscore attach. And here we need to find the actual database. This is the database. And I will run this from my root folder. It means that I will see it from data. I see data and going in there, I will go into Escal, light dash quila, do tb.

, And that is it. I will save this one. And what we'll do now, I'll close this. I don't like this. Remove that one. , Let's do DAC db quila Duck db and I will do input redirection. Go into SQL load sa sq. I run this one and it says Success, success Bullion. Several rows. , Let's.

Take a look if it's successful [00:04:00] or not. Ddb ULA ddb. Yes. Desk. Now you can say, wow, cool. I've gotten all ~~the ~~the tables that I want, right? These are all tables. And let's do, for example, from film. And you can see ah, a thousand rows. Ah, great. It works as we expect it to do. , Now I'll close this down and I will open this up in the ui.

Duck db dash ui, ula duck db.

**Kokchun Giang-4:** Since this is the first time we're working with Quila database, we will use do some EDA first. Exploratory data analysis. However, let's create a few scripts in our SQL. I open a new terminal here and I'll CD into SQL. And here I will do touch touch. Let's see, what do I want? EDA Quila S Scale.

I want to have set [00:05:00] operation under Quila S Scale. , These are what I want, and then I will go back to my D dp and I will go into EDA, and I would clear this out, make it smaller like this, and make the windows nicer to work with. ?

Yeah, something like that. , Let's start with desk here to see what we have. We have several tables here, right? Then just do like from. From actor for example, and then we'll do from film, from address to just try to understand them from category. From city, from country, from customer. .

A few of them maybe take film actor as well, just to understand it. . From actor, let's see what we have [00:06:00] here. , Here you can see we have first name, last name, last update. . And actor id. And then from. Film and you can see, , there's a film id title. Description. , Actually I'll take film actor after film because you will notice something here.

Copy this one.

Film actor is something called a association table or a bridge table. Basically it has actor ID here and it has film id, right? It has actor ID and film id. What you do is that you join together, actor ID and Film id, you will get both. Actor and film into the same table if you do the join.

This is a type of bridge table. You can see we have actor id, which corresponds to this actor ID here, and we have film id, which corresponds to this [00:07:00] film ID here. Then you can get the title together with which actor have. ~~Created which tables which has ~~participated in which titles, right?

We have that and let's continue with~~ for example, from ~~address and you can see yes, address here we have address id, we have the address, and then ~~we have continue here. Like ~~we have address two, district, city, id, et cetera. There's a lot of nos. . Just to skim through a little bit from category and you can see .

Category. We have a category ID and then we have the different names. . Nice. From. City, can this be grouped together with the address id? Maybe we have City id. The question is, ~~do, ~~does the address have city ID as well? ~~Her ~~address id we have city id. , Great. I will put them together.

City and address and probably address will have a country as well. If we take a look here, no, it doesn't have, we have city id. Let's see if~~ CT ~~id, yeah, ~~CT ~~[00:08:00] ID has country id. ~~You, ~~as you can see, you can do freeway join. I will not talk about joins much in this video. It's ~~in ~~in the joins lecture.

I will talk there. But basically you have address CT and country and you can join them together and you will get the full address right. ~~That, , ~~I won't go through all ~~of them. Let's, that is like a few ~~of them. Let's continue with something else we can do. Select, take a look into count star as number movies if we do from film, right?

Then we can do count. Distinct title as unique number of titles. , Copy this.

Let's see. Count nine, four count. Yeah, there should be a comma here. Should be a comma here [00:09:00] from this one. And you can see, yes, there are unique titles. We have thousand movies. ~~. We, ~~we'll take a look into select. I'll just copy a few here. Select Distinct Rating from Main Film. .

Let's do that. Yeah. And then we have a few ratings here. This we have done, we can do yeah, there's some more exploration we can do. , Let's type it out. Select star from, yeah. Film actor. I've already done. Let's do describe table film actor. Just do some random EVAs here.

. Film actor. Then you have actor Id film my d. Last update this one we have already ~~take ~~taken a look at. Have we taken a look at customer? Maybe we not. Let's do that from customer. And you can see, ah, this first name, last name, email, and the store id, customer [00:10:00] id. . Which store you have the bought from and which address you have, et cetera.

. This one we have here let's take a look into, can we do, for example, we can we show, ~~let's do ~~let's use some alia things. I'll show one thing. Select from customer and I will alias it as C, and then we'll use. Let's see, customer type C, first name. I'll show you what this means soon. C, last name. . Where C. First name like. Let's start with D, something like this.

I'll just copy this one and then we'll explain what it does. Run it here. . We can see customer as type, then I'll just call it type here, right? This is just to create another column. And then [00:11:00] C, first name C is the, ~~we use ~~we reference it from the alias. This is starting to we haven't done this before I think.

But the thing is now when we're using, doing compound queries, that is where we're, combining several queries together, then you should alias them that you know what column come from. What query, right from what results set. This is one query here, select these things and you have this customer sc and then you reference all the columns through C, right?

C dot using the dot notation. And here I use icle. I use where C. Do first name like D D and percentage. This will match all that will start with all the first name that will start with D, right? You can see first name do Donna, Deborah, et cetera. , These were like a few eds and there's many more eds that [00:12:00] can be been done.

More Eds left for the. Watcher or reader. If you're watching ~~this movie, this or ~~this video, then definitely you should pause and do more EDA that you understand the Quila database better. Let's move on to set operations now. It's time for doing the operations that we have learned in theory.

**Kokchun Giang-6:** ~~, we will go into the customer to select from there. ~~Let's start here. ~~Select ~~select actually I will use this here. I'll copy this one.

Here we ~~pick up, ~~pick out customer as ~~type ~~C dot first name C dot last name. From customer C, where C, first name, like I will start with A here and this will only ~~start, ~~give everything from a right. If I copy this one, you can see ~~everything that ~~starts with all all the first name that starts with a for the customer.

However, now I will union. I will [00:13:00] Union means that. ~~We, ~~we get the first results set. This is results set A, and then union results set B. Take a look into this picture here where you have union where you see the Venn diagram that. We have everything in A and everything in B, but it will remove all the duplicates, right?

~~Here select and ~~for results ~~at ~~b I'll select actor a do first name, a last name from actor. A where? A first name, like

a, ~~it's, we'll ~~start with a, I'll copy this one and now you can see, ~~ah, ~~now we get actor. ~~Customer, actor, ~~customer, [00:14:00] right? We have everything that starts with an A will be added here. If I run it again, you can see ~~it's ~~it's changing the order, right? ~~It's not ~~you can order it. Order by first name for example.

, Now it's deterministic order. What you can see here is that we have the first result set, which is this one. We get all the customer and then we union them with the actor. All the As in the actor. . This is the union operator. This will, if I give a comment here, it's selects all customers and all actors.

Let's see, selects all customers and all actors starting. First name with [00:15:00] a. , Let's continue and I will try to find the ~~over ~~overlapping names between actus and customers. To do that, we do now a little bit more space here. Find overlapping. Names between Actis and customers.

, Select from actor A. Actor A Means. Actor A. . And what do I want? I want a first name, comma a dot. Last name, right? And then I want to do intersect. Intersect means what is in common in results at A and results at B. Take a look into this Venn diagram here. Select from customer [00:16:00] C.

, Then we do C Do first name comma c dot. Last name, right? Copy this one and let's see if we find someone. There is an issue here. Table of name customers does not exist. It's called Customer. . Yes. Jennifer Davis. You are the only one that this in common in both. Just to check that this is correct, we can do like this.

We can select from actor a where a first name equals Jennifer,

or like something like this, like Jen. And you can see there's a Jennifer Davis and I will do similar thing with the customer here. Instead of a, I will do C ~~and I will do customer C. Actually, I don't need the a s ~~and you [00:17:00] can see, oh, they have Jennifer Davis in the customer as well. And also in the film we have Jennifer Davis.

, Great. Then let us find.

~~Find ~~all with initials JD and record it's type actor, ~~customer,~~

customer. That is the idea now. Let's do select. Let me give it more space here. Select from actor a let's see, and I will pick actor here as type comma a first name,

a last name. From actor A where? [00:18:00] A first name like J and a last name let's see the percentage. . And then we don't, we want to keep the duplicates. With the union, all actually there won't be any duplicates because we actually have we have a different they will all be unique from A and B.

We have actor for A and ~~customer ~~customer for results at B. , From. Customer C and here I will do customer S type and we'll do c dot first name. C, last name. . [00:19:00] Actually, since type is a result word, it's better to do like this. However, this will be small letters actually, but it's .

Where c first name like. Actually, I could copy this, right? This, because this is same. And C dot last name like D percentage. , This is our final query. Let's ~~run this one and I will ~~run this one. And you can see yes, everything that starts with J and starts with D for last name. And you can see, here we had Jennifer, ~~we have Jennifer Davis on the, ~~we have Jennifer Davis twice because we have one for actor and one for customer. And then we had some other that are actors that have the JD initials. . I hope that you've learned how to work with compound queries ~~where you, ~~where you use ~~two, ~~two or [00:20:00] more result sets together using union and intersect. Actually, you can compound more if you want. You can do another union here or an intersect as well. Set theory ~~is actually ~~is really powerful when working ~~with ~~with SQL. You use set theory, set operations to act on rows as you have seen now ~~when, ~~but we will use join later on where we will go into how to combine the columns together.

That would be quite interesting. Let's dorl d now to close this down. And if I check here, I don't have valve file, I'm fine here, but make sure you commit and push to GitHub. 

Yes. In this video we have gone through set operations using the quila database. As you've seen, the Quila database is actually quite large and extensive, we will actually work with it in several lectures. Don't worry if you haven't explored all of it because yeah, there's a lot of fun to do with the [00:21:00] Quila database.

We have worked with an EDA. But first and foremost we have been able to connect it to D db using the SQ light extension. That is quite cool that D DB can read SQ light and D DB is cool. You can read many different data formats and work it in DB and you have the power of D DB SQL.

Really nice and I hope that you learned a lot and see in the next video where we'll go into joins.

