# SQL analytics with DuckDB - introduction

[00:00:00] Hello and welcome to this lecture in SQL introduction and we'll go into the theory about the introduction to SQL and using ~~D ~~DB in SQL. This is an introductory lecture in this course SQL analytics with ~~D ~~db. You will learn a lot about how. To use ~~D ~~DB to turn metadata into valuable insights.

That is the main topic for this course. And later on in this course, we'll also connect to Python that you can work with DB within the Python ecosystem and within the data science, data engineering, data analytics ecosystem. This is really fun and we'll go directly to the slides to get an introduction to what, sQL is.

**Kokchun Giang-2:** Turning data into valuable insights using SQL. We start with an example, an ice cream startup. [00:01:00] Swedish glass is using Excel sheets to store data. This might be ~~a sound like ~~a viable solution ~~in the beginning, and it is ~~in the beginning. You just store the data in Excel sheets. We have orders, we store them in Excel sheets.

We have customers, we store them in Excel sheets, we have inventory. We store them in Excel sheets. Okay. It sounds quite simple and yeah, we can start like this. It's no problems ~~in the beginning. Okay. ~~In the beginning. But the company grew. This was really a classic, like people really like this ice cream.

They ordered a lot. And we got a lot of customers. And then when we did that, we needed to employ more people. The thing is, we have a team now, each Excel file is shared across the team. We have orders, we have customers, we have inventory. ~~And also a lot more, of course, ~~when the business is growing.

But let's simplify to this. We have [00:02:00] one person that is working with this. We have another one that is working with this. We have a third one that is working with this and a fourth one. ~~Okay. ~~If there's a lot of people working with these Excel sheets and we get a lot of data, then it's quite easy to get duplicates, for example.

Common problems that ~~are ~~arose when sharing Excel sheets in this way, we get data duplication. Accidentally ~~we create ~~creating duplicates without knowing ~~about ~~that these lines may have existed elsewhere. ~~That is ~~and that is the problem that we need to manually update in several places.

And we get inconsistent data. For example, we look for one person and then suddenly there is a similar person with different addresses or same person with different addresses. Then we have an inconsistency in addresses, for example. One is [00:03:00] a typo. Another one is not we have relationships.

It's hard to manage manually between customers and orders, which customer connects to which orders that is hard to manage manually. Team, they manually links the order to the right customer. This takes a lot of time and it's manual effort, and it will lead to inconsistencies and bugs like you, you do errors because humans it's human is not good at manual work performance.

Our Excel sheet is growing larger and larger, and this will cause performance issues. It'll get slow ~~to ~~to work with huge amounts of data in Excel and many more problems that we won't go into in this slide here. The idea is that we need something else ~~to ~~to manage this when our company's [00:04:00] growing.

An example of inconsistent data. You can see it here due to manual input. For example, here you have Alice Frost, alice@example.com and this address, and then you have Alice f alice\_f@example.com. Okay? It seems to be the same person or same phone number, same address. But here we have LSF, here we have Alice Frost.

Here are Alice F at example. Here we have Alice at example. Is it the same person or not? That is a question. Is this the same person? This is and also we will come back more into how to handle inconsistent data and how to. Make sure that the data is unique and what is determining a unique row, et cetera.

This will come back more into a data modeling course later on. However, for ~~this in ~~this course, we'll focus a lot in sk l queries and how to analyze the data. ~~And ~~this is due [00:05:00] to being able to both learn. Fundamentals in SQL and to work with it with analytics and for transactional later on.

And also we will learn it ~~for ~~that you can go~~ on ~~for example, ~~go ~~towards data warehouses and you do need to understand how to analyze the data. But that was a side note on what I just said. Now, which ~~Alice ~~is linked to what orders in the orders table. Using relational databases and SQL, we can handle many of these issues.

SQL is a structured query language ~~and we work, ~~and the database is a relational database, there are also non relational databases such as a document database, vector database, et cetera. And no SQL. But here we have relational databases and SQL, they follow something called the relational model that will come back later on as [00:06:00] well.

Define relationships in SQL tables. ~~If ~~this ensures data consistency. ~~Op, ~~they're optimized for large volumes of data, scalable and efficient for querying of data. Data constraints, like data types and unique values ~~are ~~are put in ~~that they ~~make sure that the data conforms to a specific type.

For example, you have a numeric column, then you cannot put in strings or ~~other ~~other types of data. ~~And ~~and you have unique values. You can make unique constraints. This makes sure that we get automatic validation, which reduces error. I talk about this type of thing.

This part is more concern with something called OLTP that will come back to more transactional databases. And there's a lot of ~~more more ~~nice benefits ~~with ~~with a database. SKL, what is this? [00:07:00] What is SGL is structured query language. We have create, update, and we organize data into tables.

Okay? Tables, we have rows and columns. You can see it is similar to the Excel sheets. You have rows and columns, right? And SGL has standardized language. It's standardized. There, there is a lot of different flavors of SQL, but mostly it's similar. And ~~there, ~~there's some syntax that differs.

But the foundation is the same. It's based on a relational model. And what this means is that we organize data into related tables. We link the data through unique identifiers. ~~What ~~if you don't understand what this means right now, it's fine. And we'll come back to this ~~term ~~terminology later on.

There's some core functions within SQL. [00:08:00] We get DDL data definition language.

**Kokchun Giang-3:** define structure of database, for example tables and what type of data it holds. We use the crate alter and drop, for example. We can create table, we can alter the table, we can drop tables. This is DDL, data, definition language. Moving on, ~~we have let's see. Oh, sorry. ~~We have data manipulation language, DML, these are to manipulate data directly in the database.

we have insert command, we have update command, we have delete command,

and then we have DQL data query language. It's to select to you, you use select statement to retrieve a specific information from a database. Letting users access and fill the data for insights. We use the select statement and or the select class.

We'll see. We will use it very a [00:09:00] lot in this course to analyze the data and we use it together with a lot of other things to filter. For example, the wear class, the group buy, et cetera.

And then we have DCL data control language. We have here you can revoke and grant to give access to other uses.

You, you may want to give access to certain uses for certain type, certain database or certain tables, right? Depending on which SQL you use there's a possibility for more or less. DCL. Yes. And then we go into the SQL flavor that we will use in this course we'll use Duck db.

Meet Duck db a modern, powerful database management system for analytics. It's super powerful, it's super performant. ~~And also it's something called oap. Opt, ~~it's optimized for intensive analytical [00:10:00] queries that you can do locally in your own computer. And it's very easy to set up. ~~It's it's just a, ~~you just use a ~~file, a duct ~~DB file, and then you can connect to it in the terminal.

And also we talked about Ola. Then we need to mention OLTP, which is for transactional database. That is optimized for a lot of inserts into the database. For example, other ~~SGL other ~~RD. Such as Postgres, ~~SGL, ~~Microsoft ~~sgl ~~they use and they are ~~OLCP, ~~right?

While ~~Dtb, RO ~~is Ola. It's very highly performant, on your own machine and can handle large data sets. It's embedded database, no need for separate server and database is contained in the file. This makes it very simple to work with and very good to start with as a as a first course within ~~SKL ~~analytics.

To just learn how to get insights from the data, [00:11:00] to filter the data, to manipulate it in the way that you want. And it integrates very well with other tools in the data science ecosystem or data engineering, data analytics ecosystem such as Python. Pandas and data frames. We will work with it together with pandas in this course that you can see the power of combining Python and ~~a scale combining Python and Duct ~~DB in order ~~to ~~to get data insights that you want.

Very good for data analysis, can run complex queries for analytics and reporting. And it's very good for building data transformations in an ETL pipeline to serve business intelligence and ai. We'll come back ~~in ~~in more if you follow me, ~~we have a lot of, ~~I have courses in ~~data within ~~data engineering, a lot of them.

And then there I usually build ~~pipelines ~~ETL or ELT pipelines to be more. Specific. ~~And ~~and there DDB is a core part. Could be a core [00:12:00] part of it where you store the data. You could use DDB as the data warehouse, for example. A data engineering pipeline with an OLA database as a data warehouse.

The reason why I picked data engineering pipeline is that when you're working with data analytics, you usually~~ you ~~have the data. You want the data to go through different types of transformations. Here's~~ engineering pipe, ~~data, engineering pipeline, and we use for example, DDB as the data warehouse.

But you could use something else, ~~a cloud, ~~such as snowflake, ~~for example for that but or ~~Amazon, Redshift ~~or some, ~~or Google BigQuery. But~~ but TDB ~~could work for local solution. What you have is that you have different data sources. ~~This ~~this could be like CSV files ~~here. It could have like ~~APIs.

You could have other types of data. And they are ingested into a data warehouse. The data warehouse could be this DDB file, for example. And then you serve dashboard. You [00:13:00] serve machine learning models. Then you need to transform the data ~~into certain way, ~~certain formats ~~that ~~they can be served ~~to this downstream ~~downstream.

Applications. Here you have ELT. Extract and load. You extract and load the data into your data warehouse. Then you transform it inside your data warehouse, and then you serve the data for the end users. And the dashboards and the ML models. Here, for example, other people can take over.

For example, the BI analyst could take over here could be like a machine learning engineer or a data scientist that take over and~~ and and take ~~take the data from the data warehouse. Ddb, it could work as a lightweight data warehouse for small to medium sized. Data. ~~And if you ~~and if you grow out of your DDB database that it means that you cannot do it locally anymore, then you can consider ~~cloud ~~cloud solutions.

But they ~~cost the ~~cost money. Okay? Yes.

that was an [00:14:00] introduction to ~~sq, ~~SQL and DDB ~~in ~~and kind of introduction to this course as well. But we'll have more introductions such as the core structure that you can see what is the actual content of this course. And, who this is suitable for. ~~And here but ~~I hope that you have learned some basic concepts about the motivation, about ~~the ~~why to use SQL and ~~the ~~databases and then also going into ~~like ~~DDB and some motivation about that.

Stay attuned for more videos where we'll actually go into coding and more practical development and analyzing of data. And that will be really fun. I hope to see you there and thanks ~~to what ~~for watching this video and see you in the next one ~~by, I.~~

