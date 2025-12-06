# SQL analytics course with DuckDB - subquery tutorial

[00:00:00] Hello and welcome to this video where we'll go into Subqueries in Duck db. And the subquery is a query nested inside another SQL query. And that means that it's enclosed in parenthesis, and that one will execute first resulting and the result is used by the outer query. This can for example, be used ~~it can appear ~~inside of select from where.

Or having clause. It's quite useful for building up more complex queries. But if you are building really complex queries, you might be considering CTEs which we will ~~go in and ~~teach ~~that ~~as well. There's another video where I'll cover TE however for starting point, it's good to learn about subqueries that you have more tools in your arsenals.

In this video. I will use some data that is generated with LLM. You will find that in my code repository in GitHub. Which I will leave in [00:01:00] the description of course. I will go into Visual Studio Code now and will ingest the data first.

**Kokchun Giang-1:** Here I am in Visual Studio Code, and you can see there's the students here, which has student ID grade math. This is basically the score, and we have the class names, A, B, C, D, E. This is different classes and we have different teachers that will teach different classes, ? We have class names here.

We have. Teacher name here. Now ~~we will use this, ~~we will insert this data ~~using ~~using SQL and D db. Starting with make the S-Q-L-C-D-S-Q-L, and here I will touch ingest data scale and subquery S scale. , Now you can see there's these two and we'll go into ingest data directly. For ingesting data, we'll start with creating a schema, if not exists.

And I want to have the staging schema. ~~This is where, ~~this is the [00:02:00] landing zone of the data. It's good to have a staging schema. Then we create table, if not exists staging students as. Select star from Read CSV Auto Data slash students csv. I plan to run this query from outside. From the route where I can see data and a scale folder.

. And then I will copy this one and we'll create another one that is called Staging. Teachers and I will change this to teachers as well. , Basically this is it and now I will clean this terminal and I will do cd.to jump up to the parents. And here I will do duck db. Duck db data slash I would put my duck [00:03:00] db database file inside of the data folder, like before I usually put it outside, but yeah, you can put it inside.

It depends on you how you want it. And I want, just want to show you that is possible. We can put it inside directly. Data, I will call this math. Math score Duck db and I will do input redirection. I'll go into the SQL and then ingest data. Now I have it. We'll see if I do Duck db. Data slash math score db.

Actually, I will do Dash ui. . That we open up the ui and here I can do desk directly. And you can see, yeah, we have the math score for staging with students and teachers. Great. Now let's start with Subqueries. 

**Kokchun Giang-2:** ~~Let's start with an example here. ~~What I want to do is to get all the student score. If I do, for example, ~~from ~~from staging dot [00:04:00] students, and you can see these are the student scores. However, I want to filter out those students which have higher score than the average. That is the idea.

~~I want to filter students with higher score than average. ? ~~To do that, we start with our select ~~star from, let's see, ~~star from staging students. And where, now we have this wear class. We want to have the grade math, ? This is what we want to filter on, if I can type correctly.

~~Let's see there. ~~Math, grade, math, ~~it's larger than, ~~it's larger than average, ? To get the average, we need to do like this. A subquery. We do ~~select star ~~select average of grade math from staging doc students. Students like this. Now you can [00:05:00] see we have a subquery here. ~~We can do as ~~we can call this grade math larger than ~~say, ~~average score.

Actually, the grade is strange name. Maybe it should be called score but it doesn't matter. This is the subquery. It's inside of parenthesis. This will run first. And we'll get the result , and then it'll filter on the results. And after that, we can ~~do ~~continue our queries. Like we can do order by grade math, for example.

~~I, , ~~I will stop this one and put it in here. Let's see. ~~S. ~~Syntax error at ~~near as ~~line 12. , This is not allowed. Yeah, because it's a subquery. ~~. Yes. Then I remove this one and it's correct. , ~~Now you can see that we only got, ~~here we got several, ? ~~We got 3, 4, 5, 6, 7, 8, 9, 10 rows.

While here, we only got six [00:06:00] rows, and all of them are higher than the average. . We can continue now. You can see this is the subquery. We have got out the average score to, but just to see that we've gotten the average we can take a look into from let's see, select average of grade math.

From staging those students to just see that, . 84.2, all of those are higher than 84.2? Yes, that is correct. , Let's continue with a multi row subqueries. The subquery ~~will re, ~~will return multiple rows. And this is used in it could be used in an in operator that you will have you'll check membership in, ?

Filter students in classes taught by one person. I will ~~do ~~call [00:07:00] this. Subquery that returns multiple rows used in ~~used within ~~operator to check membership. Membership

and we'll filter students in classes taught by an s. , This is the idea. What we do is select start from staging students. Where, , where class name in. And here I will do my subquery. ~~And here I will do ~~select class name. From staging teachers where teacher name [00:08:00] equals Anna s.

If you take a look into the teacher here, from staging the teachers, and you can see there's several for NS, ? Where teacher name equals Anna S. Then we'll get A and B here, ? This means we'll filter out all the class name in A comma b. We'll only get those with a comma B.

Let's run this one and we'll see. Yes. You can see we get only A and B here because those corresponds to where we have the teacher name Ana. And here ~~we have used. The, ~~we have used the teacher's table in the subquery to actually filter out the staging students. That is quite cool.

And ~~now check ~~now we'll go into something called a correlated subquery.

Correlated Subquery. The subquery, ~~it ~~[00:09:00] depends on column ~~values of outer query. Subquery depends on colon ~~values of outer query. The subquery is executed per row of the outer query, and it'll filter students obtaining math grade higher than the average of their own classes. Filter students with math grade higher than average of their own classes.

. That is the idea of this one. Let's take a look into it and then explain more. Select star from staging students. S I call it students s where ~~S grade math is larger than. . ~~S grade math. Is larger than ~~select. ~~This is the subquery. Now select average of grade math from [00:10:00] staging students where class name equals to s class name.

, This is quite interesting. It's correlated. And I would explain why. We use this. Remember what they said here. Subquery depends on column values of outer query. This is the subquery inside here. It depends on this column S class name, which is from the outside of this query. From this s table, this stage, this student's table we take a look, we find the class name and we filter it based on that, ~~?~~

~~It, ~~it says that ~~we ~~we get the average only from where class name equals esta class name. ? Interesting. What will happen now? Now we will order by s class name. . And then we take a look into these results [00:11:00] here.

? Now you can see we get for, we select all of them where we have all the students, which. Grade math are higher than those ~~in in their own in their own ~~in the average of their own class. That means that you had for example, we can take a look into from staging dot students.

We should have ~~10, ? ~~10 rows, and you have two from each. Class. And now we have only one from each class. And that is logical. 'cause we have one will be above average and one will be below average. And then now we have only those that are above average that are shown here. We have filtered out the best. Students in each class. That is what this filter, this query does. Quite cool. Quite cool. Finally I want to check the results. Check [00:12:00] results of above by looking at average grade per. Class. Take, just take a look into that. We do select class name, average of grade math as class average.

And to find this one, we'll do a group by, ? From staging students ~~group by group, if I can spell correctly, ~~group by class name. Copy this one. And we can take a look into here. Yes. And you can see that for a, we have 88.5. Yes. 92 is higher. B 83, 88 is higher. C 81 95 is higher. D 77 [00:13:00] 5. And here 81 is higher and E 91.

93 is higher. , Great. This was everything that I wanted to show you ~~in ~~in the subquery video. We'll do control D to close it down and just to take a look. We don't have a dot well file. Now you should ~~com ~~commit and push to your git and GitHub repository.

, in this video we have worked with Subqueries and we have shown different types of subqueries. We have ingested the data from the simulated data from LLM, and we have. Done some subqueries to do some different types of filtering using the wear class. You saw, for example multi rows multi column query subquery where we can use it together with the in operator that we have, checking the membership. We used a correlated subquery, for example, and then we check the results using a group by to see the actual results. I hope that you learned a lot in this video and that subqueries [00:14:00] become another tool in your arsenal ~~with ~~when you're working with SQL and DAK db.

I hope that you learned it and see you in the next video. Thank you. Bye.

