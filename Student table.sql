CREATE TABLE student2000(
	student_id int PRIMARY KEY,
	name varchar(20) unique,
	major varchar(20) not null,
	gpa decimal(3,2) default 2.00
);

SELECT * FROM student2000
INSERT INTO student2000 values(1,'Jack','Biology',3.01);
INSERT INTO student2000 values(2,'John','Maths',2.93);
INSERT INTO student2000 values(3,'Dennis','Liberal arts',3.91);
INSERT INTO student2000 values(4,'Adonis','Physics',2.35);
INSERT INTO student2000 values(5,'Mike','Engineering',3.86);
INSERT INTO student2000(student_id,major,gpa) values(8,'Arts');

select *
from student2000 
where major= 'eng'or gpa>=3.00;

-->=,<=,<>,

select *
from student2000 
where name in ('Kate','Mike','Jack');
