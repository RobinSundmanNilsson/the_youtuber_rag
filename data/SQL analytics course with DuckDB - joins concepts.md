# SQL analytics course with DuckDB - joins concepts

[00:00:00] Hello and welcome to this video where we'll go ~~into joins and going ~~into the concepts and intuition about joins that you later on can work with joins in code, which will be the next lecture. And for the coding part, we'll go into starting with some generating synthetic data to do joints to see a very simple case.

But later on we'll move on to the Quila database and do joints over there to see how that looks like. But for this one, it's just a concept. We'll move into the slides directly.

**Kokchun Giang-1:** Working with SQL, working with joints in SQL to combine columns from one or more tables. Note that we're talking about columns now except for when we're talking about cross join, which is a special case later on. But now we're talking about the columns here or we're actually talking about the columns in cross joints as well.

~~Sorry for that yes. ~~Before we talked about union or set operations and in set operations, we, we talk about rows, [00:01:00] right? Where join we are we're using compound queries. We're using several queries and compounding them together to get the, combining the results set of one with another, with a certain type of set operation.

But for joint operations, we are combining the columns, not the rows. . That is very important. Joints in SQL explain with Venn diagrams here. We use Venn diagrams similar to the set operations. You'll see that they look very similar. Here we have inner join. This corresponds to the intersect for the set operations.

But here we're using it is an intersect of the columns, right? It's the inner join here of column of table A and table B. Here, left join. You use everything that is from the left table table A and then the corresponding columns from the. B from table B there. And for the right joint we [00:02:00] pick the right table.

Table B here. And take corresponding rows for the corresponding columns for the table A and for full join, you combine all the columns. From both table A and B. Let's take a look. Here A and B are tables ~~and if, ~~and it's the columns that are joined. Let's take a look into example here.

This is an example. ~~We have, ~~we combine two tables with left join. ~~Here you have a scale code for that. You pick out the, ~~you do from main dot plans. P and left ~~the ~~join with Main Plant Care PC on a certain type of column. ~~We, ~~here, we use Plant ID and we use PC plant id. One from P and one from pc.

Here you have plant id ~~here you have planned care. The ID here, actually it's a misspelling. It should be ID here and not plant id. Yes. ~~And then afterwards, when you have done the join you pick out the columns that you want. For example, ~~here, ~~I want plant ID from ~~PI ~~want plant name from ~~PI ~~want P type, right?

I want all this from P [00:03:00] and I want PC water schedule, and I want PC sunlight. ? It'll look like this. This is the matching column. You can see. ~~And ~~for P Plant id, this ~~is, you see here ~~is the matching column with this. ~~Or actually it's not the fault I saw. I said that I didn't see a plant ID here, it's actually correct.~~

~~We have plant ID here, ~~we have plant ID here. One corresponds to one. Three corresponds to three. Four corresponds to four, right? And then ~~we have five, and ~~we have two and five ~~also, ~~but they don't exist here, right? We will have a plant name and. ~~And the ~~type ~~four, ~~two, and five. But~~ those are not ~~they don't exist in the.

In the right table, like in the planned care, which means that there will be null for water schedule and sunlight. Let's look into that. When we do left joint, and you can see they are null, right? Two and five are null. The other ones are joined according to left joint. It's ~~based on the left tables.~~

~~It's the ~~based on the left table, that is left joint. . Returns all records from left table and match with right table giving nulls in right columns if there's no [00:04:00] match. Null means basically it's missing value or not available or yeah, missing or not available. These are notes.

And here you can see, right? Join. Similarly, we have a matching column you can see here on the matching ones. But we use Right join instead and Right join. It's based on the right column. And that's why these three here become new. Returns all records from right table and match with left table.

Giving Nolls in left columns if there's no match. , These are notes and here we have Inner Join. Inner join, you can see we take those that are matching. Only one, three, and four will be matched. The rest are thrown away. That is inner Join. Intersection, right? It's the intersection of these two sets.

Returns all records where there is [00:05:00] a match in both tables. Hence no nos. And then we have full join which corresponds to the union in the set operations, right? Here we have a full join and it's this is the matching column as before, but we'll get all other plant IDs as well. But they will be nolls ~~where ~~where there are missing values, right?

Returns all records from both tables. If there's no match, nolls are used on non-matching table. Here we see that water schedule and sunlight. They don't exist for two and five. There are nulls and here the plant ID and plant name and type, they don't exist ~~in the in the ~~in the left one for yeah.

They don't exist there. Why doesn't the six exist here? Yes, because we have picked here. P plant ID and not pc plant id. That's why it's missing the six here. [00:06:00] Nos, right? These are nos as well. , Then we got come into something called cross join in S scale, L in in Venn diagrams.

It looks like this. You have all the rows here from table A and all the rows in table B. And it works like row a row row one. In table one is combined with. One and two. Row two is combined with one and two. Row three is combined with one and two, et cetera. This means that it would become something called a Cartesian product.

Let's look into that, how it looks and looks like this. If you have this the plant and you do the cross join plant and plant care, you do cross join and you'll see something like this. This is not the full list. This is not a full result. Instead you'll see here that you can see rose oak to lip, cactus, sunflower.

These are the five from the plant, and it's combined with the first one here. You can see daily. [00:07:00] And the first one here, full sun. And then next one is rose will be combined with weekly. Weekly impartial sun, and then that, rose, tulip, cactus, sunflower.

They're combined with weekly and ~~they're combined with ~~partial sun. And then later on ~~we have. ~~We have similarly here, like rose to sunflower. They're combined with biweekly and full sun, et cetera. Yes. Cross join, like right now, I just show you ~~how ~~how the result is. But there are actually quite nice application with cross joinin that for example when you want to.

UN nests un nests JSON structure json data, which actually I have~~ I have it in ~~in the appendix~~ a one appendix which is duct DB tricks, ~~where you'll see a case where I use the cross join to ~~actually unst. SKL ~~that ~~the J ~~data. That is quite cool. , Here you can see returns, ~~all records, joints all record joints.~~

All rows from left table with all rows in right [00:08:00] table. That is how cross joints work. Yes.

in this video we have gone through SQL joints. They work in db, but they work in ~~other ~~other SQL languages as well. The concepts and intuition behind them, and I hope that~~ you, ~~you have also connected it to, set operations that we learned before and the set theory that you may have learned in mathematics.

Note that in~~ SKL ~~joints, it's about the columns while the set operations in s scale, they're about the rows. These are the difference, main difference that you should know about. And then we talk about different types of joints. We had ~~like ~~inner joint, we had left joint, right joint, and we also full.

~~Full ~~join and cross joins. These were a few joins that we talked about, and ~~you ~~as you saw in the slides, ~~there were some like, depend ~~depending on which type of joins you have, ~~you get, ~~you will get ~~like ~~null values~~ when, ~~when there's missing in ~~order ~~table, right? ~~This ~~I hope that you've learned a lot in this video and see you in the next one.

Bye.

