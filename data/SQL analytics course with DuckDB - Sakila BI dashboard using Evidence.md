# SQL analytics course with DuckDB - Sakila BI dashboard using Evidence

[00:00:00] Hello and welcome to this video where we'll use evidence to create a BI report or BI dashboard. Evidence is a bi ~~IS ~~code. Business intelligence as code. We'll use Duck DB and Markdown to create SGL queries to create different statistics to create, visualization.

~~It's ~~it will create~~ a ~~a website for us and then we will be able to put in our BI analytics. And we will do analytics of the Quila database. We have worked with the Quila database in a few lectures now, and in the latest lecture we used it using ~~DLT ~~DLT we loaded the from QUILA SQ into ~~circular duct ~~db.

This ~~duct ~~DB file will put it into our. Evidence dashboard, and we will run queries to do a BI report from it. Really interesting. First of all, we'll go into browser where I can show you how to [00:01:00] work with evidence and then we'll do a setup. I'll show you how to do that, but I've already done that, of course, in my computer.

**Kokchun Giang-1:** we'll start in the browser here and we start with installing node js. And it's very important that you install version 22 ~~as ~~as of this video we recording the it evidence doesn't work on node. Version 24, but if you already have 24, it's . We can use NVM to actually change the version.

You can choose here that you want to have version 22. . Then you can see the instruction for how to install it. And you can, here, for example, for Mac Os, you can just copy ~~this ~~this command and you put it into. The terminal and then you copy this command and put into the terminal. And then you can do NBM install 22.

And then for Windows, you can see there's other commands for the PowerShell. And you [00:02:00] can have, first you install chocolate, and then you use Choco install node js, and then the particular version. And when you do node dash v, you should see the version. I'll show you in my terminal now how it looks like.

If I do node dash v, then you can see I have. Version 22.21. I also have NVM installed, you can check my version for that. It's 0 39. . But I can do like this NVM 24 or NVM use 24, I think. Yeah, exactly. Now I am in node version 24. If node dash v, you can see it's 24, but I could as easily change NVM use 22, and now I'm in node version 22.2, 1.1.

Using NVM is very simple to change the version. . I've shown you for windows and for Mac, and [00:03:00] there's also for Linux if you have that system. Just follow the instructions basically and install it. And afterwards we'll search for evidence docs. ~~Evidence you can see what is evidence.~~

Evidence is an open source framework for building data products with SQL. Things like reports, decision support, tools, and embedded dashboards. It's a code driven alternative to drag and drop BI tools. ? And you can see here to install evidence, it says to do like we needed to have node and also we need to have this vs.

Code extension. , Then ~~I will go into ~~I will go into VS code and show you how to do that.

**Kokchun Giang's video recording:** , first of all, I have this qui la b which I've taken from the last lecture where ~~we'll ha ~~we used the DLT to load the data from cul light into qui la b to show you that I can [00:04:00] open up my terminal and I will do DA db AK db. And remember that we. Put in the data into the staging schema.

If I do desk, you can see all of them are in the staging schema, and you can see some metadata for DLT. This is basically just~~ this is ~~the Quila database from the Esq light, which ~~I have ex ~~I have loaded it into the destination, which was secular Activ B. . And then I pasted it here.

Control D or control C to close it down. And now I want to do like this, I need to have an empty folder for my BI report. ~~I will make this, ~~I will do quila BI, for example, or Yeah, ~~I will, ~~I'll call it dashboard just because this is called Quila dashboard. Make their dashboard. This is an empty folder and this is what I need.

Now let's install. We'll go into extensions and we'll search for evidence.

**Kokchun Giang-2:** Yes, I searched for evidence and this is what you get. You install this [00:05:00] one you can see build polished data products with SQL and markdown. That is amazing that we can do that. Just install this one and after you have installed it, go into your folder here. And what you should do is control shift P or command shift P if you are in Mac and you ~~get ~~get up this prompt here.

And what we want to do is to type in evidence new project, , evidence new project. And let's~~ choose this dashboard. ~~Choose this folder and you can see it opened up an evidence dashboard. And I will close this down here because I ~~want, ~~actually want to be here.

And now let's~~ let's close, ~~get back my terminal. . In my ~~secular ~~dashboard, I will go into CD dashboard. Actually ~~I will ~~I will also do like this. I will create the read me here as well. ~~MD that ~~I don't want this folder to be empty, I want to say that evidence [00:06:00] dashboard will ~~ula .~~

Just to not keep it empty ~~when I will. I will move the ULA b soon. . ~~What we need to do now is I have CD into dashboard. And now what I need to do is ~~to ~~to first of all, I will go into my gi Ignore, and I will also do like this. I will gi ignore the evidence folder because here there will be a lot of different things that I don't want to put into my source control.

I will get, ignore it. And now I will do something like this. NPM install. . Remember you need to stand in the folder and you need to be working with a node version 22. ? Yes.

, Now you see it has added some stuffs, and then we also need to do NPM Run sources. And now it will install the sources. Sources here you have your database. If you take a look into how it looks like you have [00:07:00] this needful things dotdb and you have this connection yamo, which specifies to find these needful things.

Act B, you can see name needful things. Type is tb. And then option, you find the file name needful things ivb. We'll notice and we'll do a similar thing for our own dashboard. And you can see here, there's orders, select Star from orders. And then there is here pages is very important. What you need to know is about sources and pages.

The other things are. A lot of JavaScript things that I won't go into to make this work. Now when you have installed it, what you need to do is to do NPM Run Dev. ? And it'll open up a nice browser, local host, 3000.

Welcome to Evidence. This is the dashboard. Amazing. You can see that. . Here we have actually I will show [00:08:00] you side by side. I will go into index in pages. Oh, sorry. I will remove this sidebar you can see here. Title. Welcome to Evidence. Welcome to evidence. And then details how to edit this page.

And basically details, if marked on syntax, it's giving you this arrow, which you can show the details. And you can see this text is found here, this page can be found in your project at pages slash index md. Make a change to the markdown file and save it to see the changes take effect in your browser.

? Let's do that. Welcome to the EV to evidence. And this will be super fun. , Save this and you can see it appears here. Very cool. , We have here SQL categories. You can see this is just a SQL [00:09:00] query which follows~~ it follows D ~~DB syntax. It's the ~~D ~~DB dialect. You can see here.

Select category from needful things orders, group by category. And if you click on here, you can see this is the categories that we get. We get these four categories, and then it uses these categories. You can see it's called categories, ? It uses it in the data, you can see dropdown data categories.

That's why our dropdown here, I won't go into exactly details how this works. But you can see these categories appear here. Cast sporting, this one, mysterious apparel, this one odd equipment. This one seen as the toy. This one. . And then if you pick different categories this is quite interesting because if you pick another category, then the categories will be used because remember, the name is name is category.

, Name is categories. Then [00:10:00] we have here an SQL, which is called orders by category. And ~~you can put it in here, ~~you can put in this value here. Where category, like in pit, stock category dot value. Basically if I choose another one, curve Sporting, you can see, oh, it changes. The data changes, ?

And quite interesting is that the graph changes also. , There's a bar chart. And the bar chart picks the data from these orders by category. And we have title, we have X and Y, and then we have series. , If I choose another year, 2019, you can see, ah, 2019, 2020, ah, it changes. That is cool. And ~~it ~~you can see ~~how ~~how nicely it changes as well.

~~That, ~~that is super cool. Basically this is your starting point. And you have based on this index, we will take out the things that are interesting for us to create our own ULA dashboard. In order to do that, let's [00:11:00] create another pages here. Just create a new file called Sula mnd.

And now you can see there is another page called sula. , This is a hamburger menu, but if you enlarge it, you can see it directly here. It's also responsive. If I make it smaller, we need to click it to find ~~ula. , ~~ULA. Now you can see it's empty, but we have this breadcrumb navigation.

Then we can do similar thing as before. Let us create our ula. Actually we need to do another thing. We need to put in some sources because I want to have the database. I will create, I will mark sources here and create a new folder that they become siblings. And I will call this let's see.

And inside there, I will drag in my Ula Tob, move this one. You now you can see that there is some sources that has been put in [00:12:00] and we need to, I will copy this connection as well. And here I will change the connection to name is Not Needful things, it's ULA tb. And then we have ULA tb. Save this one.

~~And I will also do. Let's see. I will also do,~~

for example, I can do, I can also do I will create another~~ I will create a a scale file film that, a ~~scale for example, and I can do something in style from staging film.

If I do this and I can do, yeah, I will do this one and then afterwards I will go into pages and I will write out my code. I'll call this exploring ULA database. Save this one. Ah, you can see ~~it's ~~it starts out here ~~out ~~directly and then I will do for example, films [00:13:00] in ula. This is the second header.

~~I actually want to, this disturbs me a lot. I will remove my. ~~Markdown. I removed the GitHub copilot.

, Now I want to pick out, I would want to do an SL. I'll call this films.

These are B, and let's do like this. I want to select a few things from film. First of all, our film is from ULA film, ? This is where our data is. If I save this one, you can see this is our data. And you can see the compile. The S scale is like this from Ula film. And we got thousand records with 15 properties.

Now you can just scroll it and you can see the data. But I want to pick out this. When you do ULA film, it's based on here, you can see the sources. You have ula and you have this film Do [00:14:00] scale. And in this film do scale. It'll connect to the ULA activity. That is how it works. If I, for example, did actor, then it should not work because I don't have an actor, ?

I don't have this table. That Akila film I have because I have done this film from staging of film. ,

Based on this one, I can do select, I can pick out the things I want, title, for example, the description, for example, what do I want more? I want rating. I want to have length. Oh, sorry, length. I want to have release here, for example. Save this one. And now you can see, ah, now I only got these five properties or these five columns, ?

And then I could move [00:15:00] on. I will actually do like this. I want to have. I want to have another SKL file here, I'll create another one. I'll call this rental customer S Scale.

**Kokchun Giang-3:** the rental customer will pick out a few things that I want. If you know about, let's take a look into ERD. This is the ERD and the rental customer. Basically, ~~we, ~~what I want is I want this rental. I want to have this customer, I want to join the customer rental. And I want to join the payment.

And also I want to join which store this is. Let's see, customer to store, ? Store customer rental and payment. The reason for that is that I want to be able to later on calculate some metrics on how much the total payment has been for each [00:16:00] store. To do that, I want to pick out a few.

I want to pick out these four that I can actually calculate this out. In order to do that, we need to join them together. ? I will start with the payment and I will ~~join with the ~~join with the rental, and then I will join with the customer, and I will join with the store. Then I get these four.

? To do that, we'll do select. Select from, actually, let's see, select from staging payment P And here I want to do left join staging rental ~~are on, ~~on P rental ID equals ~~r ~~rental id. And let's see, yeah, I will actually make this a little bit [00:17:00] larger like this. . And I will start with picking out the columns that I want.

I want to have P payment, LP amount. I want to have p payment date. I want to have r rental date. ? And I also want r return date. ? This now you can see that if I do see my colon here and I have rental customer and I will pick the rental customer. I'll go into Quila here and ~~I will do ~~I will do.

I will call it top customer first, top customer.

What I want to do is I want to have

S rental. And I start with from Saki Rental customer. Now I can see top customer

and you can see the columns that I picked out. [00:18:00] And I will continue to pick out some columns now. We'll continue the joining. Here I will copy this one left. Join staging dot customer. See on let's see. On C, customer ID equals to p customer id. . And then I also want to join the store staging.

Do store let's see, store. S on S, store id. C store id. . And now I want to pick out some more things I want to pick out s store Id. I want to pick out what else do I want to pick out? I want to pick out C dot customer id. I want to pick out C first. Name [00:19:00] and then concatenate it with its last name.

C, last name as customer. . Save this one. Actually, I need to have a comma here. Now you can see, yeah, we get the customer. We have amount payment we have payment date, rental date, return date, customer ID customer, which customer it is, and which store ID we have, and we have customer ID as well.

? Whatever you want, you can pick out from this scale queries, and then afterwards you can use this rental customer ~~inside of our inside of our pages, ? ~~Inside of the Quila pages. Now moving back to secular pages. I want to do a few things here. First ~~I want to have ~~I want to pick out the top customers basically.

What we want to do is from ~~t ~~rental [00:20:00] customer, I want to do select customer comma some of amount as total amount. And I want to pick out count star as. Number of rentals. These are my top customers, and I want to do that. Then I need to do group by customer id, comma, customer.

I want to pick out~~ both, ~~both the customer ID and the customer. I group them by, and then I will do order by total amount. And descending common number of rentals descending as well. And let's do limit 10. , Top 10. And now you can see my top 10 customers. We have customer total amount, number of rentals.

That is quite cool. Now I've picked out my top customers based on [00:21:00] this query, and we could do a bar chart based on this one. Let's. Pick out bar chart. This is the code for bar chart. What you could do is that you let me see there, they had a bar chart here, in index.

You could just copy this one. This is the bar chart, and we could just paste this in, but we change the things we want. Data, it's not orders by category. Instead, it's this rental here, ? It's called rental. I'll just put in rental and then X. What is that? X, we can choose what we want. For example, I want to have a customer as X, and then total amount as Y.

Total amount out series. Let's see. Actually, I don't need this one. And I want to have a title [00:22:00] as well.

The title, top Customers Measured by Total Amount Paid. Save this one. And now you can see I get this nice graph here and you can also see that it's if I hover over it. You can see that we can see the values here, that is quite cool. And I can also do swap XY equals to two. Save this one, and now I get a horizontal bar chart, that is quite cool.

Let's continue here. ~~I will do. ~~I will I'll just do like this to pick out all the data. SKL, I'll call it Rent from ELA Rental Customer. . Save this one, and now you can see the data is coming out here. And now I [00:23:00] want to do some analyzing. Analyzing payments. . I can say choose your store and then I will have a dropdown here, drop down.

And as simply, I could just copy it ~~from ~~from the code before. But ~~let's ~~I can write it out here for you. Drop down. And here we need to pick the data. Data equals to rent. I picked the data from ~~this this. ~~This SQL query. We have all the data from there, and then what I can take is name equals to store value.

This is how I will refer to it later on. Value equals to store. Id remember that it has a store ID somewhere. Let's see. Yeah, if we continue here, we see the store id, ? The value comes from the store Id. And we can have a title equals to select a [00:24:00] store and we have a default value equals to one.

, Save this one. And now you can see we have a dropdown, we can pick one and two. , What to do with this? We can do something like this. We can say like a store. You have the store and then we can do calibrates inputs dot, it's called the name store value. Now you see you have picked store two.

If I picked store one here, you can see we have picked store one. . And then for this one, let's create a SQL query now that we will have, the data shall differ depending on which store you pick. We do scale customer.

I call this [00:25:00] customer I can do select and then from ELA rental customer, and I want to select here date trunk. I want to truncate it into today for the dates. And then I want to pick the, if you see like the payment date here, actually we, you can see it as date, but actually there, there could be like.

All the numbers could be like a time or . This one, make sure that this tran is in today, payment date as pay date, and then. Since I have it as pay date, then I can group it by, ? I could do group by pay date, and if I group it by, then I can do some of amount. This is my aggregation function, total amount.

[00:26:00] What I want is to have. Total amount based on the store, but actually I don't have the store yet. We need to do where store ID equals to. Now I pick out, here I pick out the inputs, do store, do value and payment date. ~~I will have ~~I will have full payment date. I will have 2000. Five zero. This, you should know if you do some EDA and understand the data set better, but how, which ranges you can pick.

2005 to 0 9 0 1. The idea is that I want to have now you can see the data. We can have, for each date we get different total amount, ? Each date we have different total amount. That is what I've done. I have basically, and also if you see, if I change this one to one, you can see [00:27:00] the value has changed.

Two, the value has changed. One, the value has changed because for we have this wear clause here. Depending on which store we have, you can see now it's one. If I change it to two, you can see it's two, ? Depending on which one it is, then we filter the data, ? We filter out the rows that we want, and then we use that into our aggregation.

That is how it works. And now when we have this one, we can simply do something called a line chart. This is quite cool. A line chart will have data equals to a customer because remember that I call this one customer, this data here, and then we have x equals to pay date, for example. This is the payment date, this date here.

And then y equals to total amount. Amount, [00:28:00] and we have x axis. Title equals to total amount. We have Y axis. This title equals to payment date. And we have the title equals to payment over time for store. And here you could pick out inputs.store value, for example, ? Payment over time for store two i, I will just collapse this one and I'll pick another store.

Now you can see store one and you can see the graph has changed a little bit. You see the graph is changing. That is really cool. It changes in real time. Yes. That is what I wanted to show you in [00:29:00] this some very simple ula I'll show you here full screen how it looks like. What we've done is.

We have some header, headline headers and subheader. We have some scale query. We get the data. We have another s Scale query, and here we get data. We use this data to draw this graph, and we have another scale query here, rent. And then based on this one we we can, we can, we create this dropdown and based on this dropdown, you can see, depending on which one I choose, this one changes.

And also the data changes because it's based on the value from the dropdown and also this. Graph changes because it's based on the data from this query here. This is really cool. And now I have quila. I also have the home if you, this is the original that is here. I still have it here.

I haven't done anything. But ~~from here. ~~Basically, if you want to have this as a PDF report, [00:30:00] you do command P or control P, and then you can just save it. Say this one. We can call it Quila BI Dashboard. Quila BI Dashboard. And you can see this is the dashboard that we have. Actually this one didn't show because actually you need to choose it first, I think.

Yeah. Let's yeah, we need to choose a store. If I choose this one and now you can save it again. Quila BI dashboard. Yeah, replace that one. Quila BI dashboard. And you can see yeah, here is the graph. That is quite cool. Now you have a BI report that you can just send to anyone you want. Very cool.

Make sure to commit and push your changes to Git and GitHub.

Cool. Now, ~~we're, man, ~~we [00:31:00] managed to make a dashboard based on just markdown and sgl. You don't need to use Python, you don't need to use other~~ languages to ~~programming languages to make a dashboard. Just mark on SL. Inside of evidence. That was really cool that we did that.

And as you saw, it wasn't very hard to do. Maybe the setup. The setup is also quite easy if it works. But maybe there's some things that may not work that you need to debug. But after it has been set up and you learn how to create a dashboard based on this one, basically you just take the things in the index and~~ and just adapt it to your, ~~adapt it to whatever you need. And if enough scale, you have a lot of flexibility to, pick out the data. You want to group the data, you want to filter whatever you want and then you can create charts out of it. And also, if you need more things, take a look into the ~~document ~~documentation.

It's really good. And there's a lot of things that you can do with very simple syntax. With this, I hope that you've learned a lot in this video, and thank you for watching this [00:32:00] video. See you. Bye.

