# SQL analytics course with DuckDB - joins with sakila database tutorial

[00:00:00] Hello and welcome to this video where we'll go into the quila database, and here we'll use the Quila database for joint operations. We'll try out the different joint operations that you've learned in the earlier lectures. Before we have also worked with the Quila database in. Using set operations.

I won't repeat the exploratory data analysis over there. Instead, if you want to know more about the database you should take a look into that video where I do an exploratory data analysis. And afterwards I will, I go into the set operations. Now we will I will, however, show you how to set up the Quila database into ddb, and then we'll work directly into join operations in Quila database.

let's move on to, first the browser to ~~see ~~see the Kaggle and see where you can get the database. And also I will leave this link in the description, of [00:01:00] course. .

**Kokchun Giang-1:** Here I am in the browser into Kaggle, and you should just search for SQ light QUILA sample database and you will come to this page here. I will also leave the link in the description that you can find this dataset. And basically this is a sample database of a fix database to represent a DVD rental store.

It was originally created for my sq l. And then afterwards people have used it for different types of SKL engines as well. And for this one is SK L Light. This is our starting point. We'll download this one. You just go up here and click on download.

And you will get a ar an archive zip, and then you unzip it and move it to a Visual Studio code. And then I will show you how to get this data into ddb.

**Kokchun Giang-2:** I am in Visual Studio Code, and you can see that I have copied in the data. I copied in the whole folder after I've unzipped [00:02:00] it. I have a data folder, and inside of that I have this escal light dash db. And this is basically SGL files to ~~generate this ~~generate this Aquila database.

However, I will just delete this one as I don't need that one. And read me no text. It's about the Quila database. And then we have here is the ERD diagram, entity relationship diagram. And you ~~can take us a quick look in, we ~~can take a quick look into that. We will focus on the film and film actor and actor and film actor ~~is a ~~is a bridge table.

You can see there's. Foreign keys that is referencing the primary key in actor and in film. This one has both of ~~the, ~~those IDs you know, which actor is corresponding to which films. That is how it works, and ~~we will, ~~we will be able to join them together soon. . And let me see if there's something else.

And also. Let's see. Film actor. Yeah, [00:03:00] exactly. And also I will go into film category where film category is ~~also a it's ~~also a bridge table. You can see here's foreign key to the film id and here's the category id, foreign key to the category. , That you can join them together to get both the name of the category ~~and the film, right, which, ~~and the film title basically.

These are a few that I will take a look at and I will also show later on a freeway join where we'll join into ~~address. And here, where's address? Here's ~~address. And in address. You see. There's a CT id, a CT ID is, is connected to the city here, and the CT ID is in turn connected to a country, right?

Here we can do a freeway join to show this one. This will be quite interesting, we won't work with everything in the Quila database. There's a lot here. We'll just show. Some small parts of it. Let's, you see, you have Esq Light Dash Quila DB [00:04:00] here. This is~~ the, ~~the Esq Light.

This is a file database file that is for Esq Light. But we will work with D db. What we'll do is that we will attach the D db the ES scale light extension to D db, and then we'll load this database into a D DB database. And when we're, we have all the tables in the D DB database, we are home.

We know how to work with it, right? Let's start with your terminal now. Here I am in the terminal and I will do make the s scale. And inside I will jump into S scale. And inside of S scale, I will create a few files. I will create join I will create load saki levers dot scale. And then I'll create join ULA S scale.

These are the two files I have. Ls, you can see these two files here and I'll see the.dot to jump [00:05:00] out. I want to be in the root folder that if I do ls, you can see that I see data and S scale. This is where I want to be. And what I want to do is to jump into load ela. This is our starting point.

Install Esq Light. I want to install this extension, and then I want to do load Esq light. After I install it, I want to load it, and then I want to call Esq Light attach. I call this function. And I want to attach it into data slash eska light dash ULA db. I go into data, I want to go, go into this ESQ Lights dash ULA db, and I want to do, see my colon here and this one.

, Now it's time to create a duck DB database. Duck [00:06:00] db I want to actually put it into data. And I call it escal light dash not escal light. I call it ULA Duck db. And I want to do input redirection from SQL. I want to do a load SLA sq. , Success. Now I have it inside of my you can see there's a TB there.

What I want to do, stack db dash UI data slash. Ula. , This is the ui. Like we, I, I will show you the UI soon, but here I will start with desk to see that it works. , It works. You see we have 25 rows. That means we have a minus these four, we have ~~20 wine, ~~21. ULA tables. 21 ULA tables.

That is quite cool. Let us do, for example, from film, and you can see [00:07:00] this is our film. ~~From film limit five maybe. Yeah, exactly. ~~And you can see these are the first five of our film table. 

**Kokchun Giang-3:** let's take a look into for example, I will start actually in the terminal here and from film, and you can see I will do from Film Limit five.

~~Yeah. Actually, . . Like this. ~~And I will do from Film Actor Limit Five.

And you can see here we have Actor ID and Film id. Here we have film ID and we'll do from Actor Limit five.

And you can see the actor ID here. And we have first name, last name, right. And here we have title and we have some other things. ? . Let's do a select Now. I will change to join [00:08:00] secular. We'll make a join now. Select from. Actor a left join. Starting with a left join. Left join. I want the left.

Join film actor FA on. Fa Actor ID right equals to a actor Id. . And now I want, what do I want to select? Well, I want to select actor id. I want to select a first name. I want to select a last name, and I want to select f a.film id. I want to know which. Film Id corresponds to, and as you know we, we need to use the Bridge Table film Actor in order to join the film [00:09:00] id, and then we need to join again to the film table that we can get the actual film title right.

This is a good starting point. I will copy this one.

Basically what we want to find here is. Actually, I'm using left joint, but I will explain what I'm looking for. , Which actor has played which film id? This query answers this question right?

Now we know which actors have played which film, maybe. And of course~~ it's not, ~~it's very common that an actor like Penelope Guinness have played in several films, right? And then we have Nick that had played in several films as well. . Using this, you could, for example continue with doing different grouping and actually ~~ca ~~calculating or getting some descriptive statistics on ~~like ~~how many films ~~each, ~~[00:10:00] each actor have played on, for example.

. But ~~I, ~~I won't do that right now. Instead, I will focus on the join. I will continue here with I want to find out, I'm not satisfied with this one, but ~~I will ~~I will actually copy all of this. Now, which actor has played which films? This is what I'm looking for now. What I need to do is to actually join Free Tables.

~~Join free tables or free way joins. , ~~Now we have joined that we get the id, the actor ID and the film id. Right? I want to go in and actually join on the film. One more time left. Join. And what do I want to join on? I want to join on film F on. F film ID equals to FA film id. That is what I want to do.

And now I don't need to [00:11:00] go into film ID instead I can directly pick F title, right? And I can remove a actor ID as well. I don't need that. Now I have, let's see if this one will work. , It works. Now I can see Pan Guinness have played. All these titles here. . Cool. Let's continue with, ~~when, ~~when we're doing left join, we get all films, including those without a category, and the category becomes null basically.

If we have a category. While the inner join, it'll get all films with associated category. ~~That, ~~that is what ~~I will be looking for. ~~I will be looking for I want to get all films with associated category. And for doing this, I need to go into the film category [00:12:00] now. ~~Select ~~select from, film category, FC Inner Join category C on FC dot category ID equals to C category id.

, Starting with this, we'll do F title. Actually, we don't have F yet, I want to get c.name as category.

And ~~I want to do ~~I want to do another inner join. Inner join now it's time to get into film F on. F film ID equals to FC film id. , I will explain this more detail now. F title. . Copy [00:13:00] this first. Run it here. How does this work? Well, it works in the way that. ~~We start to, ~~we start to take from film category, which has ID for film and film ID and category id.

Right? And for that we will inner joinin the category ID that we get all the categories, which are like documentary horror, et cetera. And we do inner join with the film on the film ID that we can get the title right. And idea with inner join is that ~~the, ~~this, it's the intersection, right? ~~Of ~~of this.

It means that we get all the films with the associated categories. If we would be doing Left join, then we would get all the films. Including those without categories. It could be like films where we don't have a category, those will become null, but when we are doing inner joint, we won't get any nulls.

That is the idea.

**Kokchun Giang-4:** let's test something [00:14:00] else. We want to see a cross join in action. Let's start with selecting count, star. From film. I want to show you, ~~I, ~~I can set, send it down here and we can see 1000 and I want to copy this one and I want to check count star of category. . Copy this one into here and you can say 16.

If I do cross join here, I should have 16,000 rows. Cross join, you've seen in the theory that we take one row in one table ~~and join ~~and join it with every other row in the other table, right? Let's copy this one and now let's do a cross join and I will show you a quick way to do that.

Here we have select count star from film F Inner ~~join. In the ~~join main dot [00:15:00] category. ~~Or ~~on ~~the categories, enough ~~category C on category id. . Note here that we have, if we do desk of table category, and of course this one has category id, but if we do desk table film, you can see that it doesn't have a category id, since it doesn't have a category id, when I do it like this, it will automatically do a cross join.

It will join we have 16 rows for the categories and we have thousand rows for the film. It will be 16 rows, 16,000 rows here. I do this and you can see there's a 16,000 here. This is actually doing a cross join. You can take a look into star here and you can see that. ~~, We have here, let's see, actually I don't need to pick everything.~~

I will pick f title and we'll have category C dot what did I have as name for category? Category, just only, I think. Yeah. The table [00:16:00] name the column name~~ category C does not have a colon name category. , Let's see, what did I pick here? Film category ~~c.name as category. All right.

c.name. See that name as category. . And you can see there's like action on all of these. And then when action is finished, like 1000 of actions. We should go into 2000 and you can see there's a lot of children until we get to 3000 something. Yeah. Continue children. And 3000 is the classics, right?

That, that was a cross join. , Let's try to join on some addresses. Take a look into, in which address city and country does the staff live in? , Before answering this we need to take a look into the staff, right? We will do. Here from I will do some from [00:17:00] staff from address, from city.

We, we took a look into the ERD before and we saw that this was possible. To answer this question we see country, ? I will take a look into staff, how that looks like. Let's take a look into staff. You can see here we have. Staff ID one and two. Where Mike and John from address we have, what do we have here?

Now? We have a lot of addresses. We can actually take a look into desk of table address and you can see there's address ID address. Address two address CT id, right? There's a city ID here, what I should take a note here is in the address we have. What IDs do we have? Address id, we have address and we have CT id, right?

And then in the city we have let's see, desk, table, ct. [00:18:00] We have ~~C ~~City id, we have city, we have country id.

And then we have for country, let's take a look into that from country and we get. Country and country id. Country ID and country. . This is~~ this ~~fast exploration as to from staff as well. It can take a look into what we have here. We have address id, we have a first name, and if we have first name, then we'll probably have last name as well.

. And password stored like this. Yeah, but this probably hashed. . No matter. . Let's do some joining. First of all we will do select from staff s left join. From staff s and then we do [00:19:00] left joints and address a on s dot address ID equals to a address id. . We saw that address, have address ID and staff have address id.

Then ~~we have, ~~we get into address id, we can do, ~~let's ~~let's take a look into star here and just copy this one. Let's see. Address id, yeah. Two s.

Like this. , Take a look into this one and you can see, ah, we have a first name, we have last name, we have address id. We have email, et cetera, store ID and other things. And we have the address. ? Cool, and we can continue here. What else do we want to put in? We want to put in join. Let's see.

We have address, we can go into the city as well. ~~C ~~on address ~~a address id e a CT ~~id equals to~~ CT c, ~~ct id.

[00:20:00] And then we can continue left. Join let's see. We go into city and we want to find a country, right? ~~Country city, CT y. . ~~Country on ~~CT YC. ~~Country ID equals to ~~c ~~country id. . And now we want to pick out things that are interesting for us. We want to pick out the first name, last name, address.

. First name comes from S, right? First name~~ s ~~last name, and then we have a address. We have ~~c ~~city, we have~~ CT Y ~~country, right? Now we have our query and you can see it works. We get Mike and John ~~it ~~with this address and living in these cities and this country. That was quite neat.

We did a freeway join and note one thing [00:21:00] that we have put the join in this order, but if you change the order, it doesn't matter. The join order doesn't matter. We were able to join several tables and you can join more tables if you want. For example, you saw that the staff also had like store ID and we could join to the store and get the store name, maybe the location of the store as well.

There's a lot of things, but I won't take away all the fun for you. Instead, you will. Get to work with this in exercises and also do your own explorations, whatever you want to get as data. What I do now is that I will close this down, control D and the duct DB is closed down and you see there's no dot valve file 'cause we haven't modified anything in the database.

I'm satisfied here. Make sure you commit and push your changes to Git and GitHub.

. In this video you have, we have gone through joints in the Quila database. We started with going into the quila. Database that was SQ light in the beginning in Kaggle. And then downloaded it and put it [00:22:00] into the S code. And then creating the duct DB and loading it from the SQ light using the SQ light extension in duct db.

And then we did some joint operations. For example, left Join. We did inner join. We did cross join to show that as well. And we did like free J free way joins to see that we could join to get us more tables, to get more interesting columns. As this Quida database is highly normalized.

I think it's free enough, I'm not sure. But later on you will be able to, take a look into that yourself when you go into data modeling and learn more about the normalization on what that means. , But I hope that you've learned a lot using joins in the Quila database here. Thank you for watching this video and see you in the next one by.

