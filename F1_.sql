-- 1. Вывести количество треков по странам --

SELECT country, COUNT(location) AS 'Количество'   
  FROM circuits
 GROUP by country
 ORDER BY 2 DESC;

-- 2. Вывести кубок конструкторов за 20-23 годы. Сортировка по убыванию --

SELECT con.name, 
       SUM(rc.points), 
       r.year 
  FROM results rc
       INNER JOIN constructors con USING (constructorId)
             JOIN races r USING (raceId)
 WHERE r.YEAR >= 2020
 GROUP BY 1,3
 ORDER BY 3, 2 DESC;

-- /SELECT r.raceId, r.name,
	-- COUNT(CASE WHEN r.year = 2020 THEN r.year ELSE 0 END) AS '2020',
	-- COUNT(CASE WHEN r.year = 2021 THEN r.year ELSE 0 END) AS '2021',
	--  COUNT(CASE WHEN r.year = 2022 THEN r.year ELSE 0 END) AS '2022',
	-- COUNT(CASE WHEN r.year = 2023 THEN r.year ELSE 0 END) AS '2023'
  -- FROM races r
 -- WHERE r.year >= 2020
 -- GROUP BY 1,2 -- 

-- 3. Вывести гонщиков с наибольшим количеством побед. Сортировка по убыванию. Вывод ограничить 20 пилотами --

SELECT CONCAT(d.forename,' ',d.surname) AS "Pilot", COUNT(r.YEAR) AS "Количество побед"
  FROM results rc
   		INNER JOIN drivers d USING (driverId)
   		      JOIN races r USING (raceId)
 WHERE r.YEAR >= 1950 AND rc.position = 1
 GROUP BY 1
 ORDER BY 2 DESC
 LIMIT 20;
 
-- 4. Вывести гонщиков Ferrari, выступавших за конюшню с 1996 по 2023 годы, и посчитать их набранные очки. Присвоить им ранг.

SELECT DENSE_RANK() OVER (ORDER BY SUM(r.points) DESC) AS 'Rank',
	   CONCAT(d.forename,' ',d.surname) AS 'Пилот',
  	   SUM(r.points) AS 'Очки'
  FROM drivers d
       INNER JOIN results r USING (driverId)
       	     JOIN constructors cons USING (constructorId)
       	     JOIN races USING (raceId)
 WHERE cons.name = 'Ferrari' 
   AND races.YEAR >= 1996 
   AND races.YEAR < 2024
 GROUP BY 2
 ORDER BY 1;

 -- 5. Вывести гонщиков Mclaren, выступавших за конюшню с 1996 по 2023 годы, и посчитать их победы и набранные очки за эту команду. 
 
WITH cte AS(   
SELECT CONCAT(d.forename,' ', d.surname) AS "Пилот",
       driverid,
       SUM(res.position) AS "Победы"
  FROM results res
       INNER JOIN drivers d USING (driverId)
			 JOIN constructors cons USING (constructorId)
			 JOIN races USING (raceId)
 WHERE cons.name = 'McLaren'
   AND races.YEAR >= 1996 
   AND races.YEAR < 2024
   AND res.position = '1'
 GROUP BY 1,2
 )
          
 SELECT Пилот,
        Победы,
        SUM(results.points) AS "Очки"
   FROM cte
        INNER JOIN results USING (driverId)
              JOIN constructors cons USING (constructorId)
  WHERE cons.name = 'McLaren'
  GROUP BY 1,2
  ORDER BY 3 DESC;
       	     
-- 6. Вывести 15 лучших кругов на СПА. Присвоить ранги. (res rank)

SELECT DENSE_RANK() OVER (ORDER BY milliseconds ASC) AS 'Rank',
       r.YEAR AS 'Год',
 	   c.name AS 'Трасса', 
 	   time_format(SEC_TO_TIME(milliseconds/1000), '%i:%s:%f') AS 'Время круга, в мс'
  FROM laptimes l
	   INNER JOIN races r USING (raceId)
             JOIN circuits c USING (circuitId)
 WHERE c.name = 'Circuit de Spa-Francorchamps'
   AND milliseconds IN (
                        SELECT min(milliseconds)
                          FROM laptimes l2
                               INNER JOIN races r2 USING (raceId)
                                     JOIN circuits c2 USING (circuitId)
                         WHERE c2.name = 'Circuit de Spa-Francorchamps'
                         GROUP BY r2.YEAR, r2.raceId, c2.name
                         )
 LIMIT 15;

-- 6.1. Вывести 15 лучших кругов Имолы по годам.
	
SELECT DISTINCT r2.YEAR,
                c2.name,
                MIN(time_format(SEC_TO_TIME(milliseconds/1000), '%i:%s:%f')) OVER (PARTITION BY r2.YEAR) AS 'Время, в мс'
  FROM laptimes l2
       INNER JOIN races r2 USING (raceId)
             JOIN circuits c2 USING (circuitId)
  WHERE c2.name = 'Autodromo Enzo e Dino Ferrari'
ORDER BY 3 ASC
LIMIT 15;

-- 7. Вывести трассы, на которых побеждал Хэмилтон. Добавить столбец накопленного итога

WITH cte AS( 
SELECT c2.name,
       SUM(res.position) AS "Победы"        
  FROM results res
       INNER JOIN drivers d USING (driverId)
			 JOIN constructors cons USING (constructorId)
			 JOIN races r2 USING (raceId)
			 JOIN circuits c2 USING (circuitId)
 WHERE d.surname = 'Hamilton'
   AND d.forename = 'Lewis'
   AND res.position = '1'
 GROUP BY 1
)

SELECT c3.name AS "Трасса", 
       Победы,
       SUM(Победы) OVER (ORDER BY c3.name) AS "Накопленный итог"
   FROM cte
       INNER JOIN circuits c3 ON cte.name = c3.name;

-- 8. ТОП-10 трасс, на которых Ferrari набрала больше всего очков с 1996 по 2024
       
SELECT c2.name AS "Трасса",
       SUM(res.points) AS "Очки"        
  FROM results res
       INNER JOIN drivers d USING (driverId)
			 JOIN constructors cons USING (constructorId)
			 JOIN races r2 USING (raceId)
			 JOIN circuits c2 USING (circuitId)
 WHERE cons.name = 'Ferrari'
 GROUP BY 1 
 ORDER BY 2 DESC 
 LIMIT 10;
 
 -- 9. Позиции команд в КК
 
 SELECT DENSE_RANK() OVER (PARTITION BY r.YEAR ORDER BY SUM(points) DESC) AS "R",
	    con.name,
	    r.YEAR       
   FROM results rc
	    INNER JOIN constructors con USING (constructorId)
	          JOIN races r USING (raceId)
  WHERE YEAR >= 2000
	AND YEAR <= 2023
  GROUP BY 2,3;

-- 10. Сравнить выступления Ferrari с 2000 по 2023 годы (сравниваем очки)
  
WITH prev AS(
SELECT con2.name AS "con_prev",
	   r2.YEAR AS "year_prev",
	   SUM(rc.points) AS "prev_pts",
	   LAG(SUM(rc.points), 1) OVER (ORDER BY YEAR) AS "prev"
  FROM results rc
	    INNER JOIN constructors con2 USING (constructorId)
	          JOIN races r2 USING (raceId)
  WHERE YEAR >= 2010
	AND YEAR <= 2023
	AND con2.name = 'Ferrari'
  GROUP BY 1,2
)
	 
SELECT con_prev AS "Команда",
       year_prev,
	   prev_pts,
	   round((prev_pts - prev)*100/ prev, 2) as "diff"
  FROM prev;
 
-- 11. Соотнести по командам гонщиков по их количеству побед (ранжировать их в рамках команды)
   
WITH cte AS(   
SELECT CONCAT(d.forename,' ', d.surname) AS "Пилот",
       cons.name AS "Команда",
       driverId,
       SUM(res.position) AS "Победы"
  FROM results res
       INNER JOIN drivers d USING (driverId)
			 JOIN constructors cons USING (constructorId)
			 JOIN races USING (raceId)
 WHERE races.YEAR >= 1980 
   AND races.YEAR < 2024
   AND res.position = '1'
 GROUP BY 2,1,3
 )
 
 SELECT DENSE_RANK() OVER (PARTITION BY Команда ORDER BY Победы DESC) AS "R",
 		Пилот,
 		Команда,
        Победы,
        max(Победы) over (partition by Команда) as max,
        min(Победы) over (partition by Команда) as min
   FROM cte
  ORDER BY 3, 4 DESC; 
	
-- 12. Какой процент от всех побед конюшни внес гонщик

WITH cte AS(   
SELECT CONCAT(d.forename,' ', d.surname) AS "Пилот",
       cons.name AS "Команда",
       driverId,
       SUM(res.position) AS "Победы"
  FROM results res
       INNER JOIN drivers d USING (driverId)
			 JOIN constructors cons USING (constructorId)
			 JOIN races USING (raceId)
 WHERE races.YEAR >= 1980 
   AND races.YEAR < 2024
   AND res.position = '1'
 GROUP BY 2,1,3
 )
 
 SELECT Пилот,
 		Команда,
        Победы,
        SUM(Победы) OVER (PARTITION BY Команда) as "Общее число побед",
        ROUND(Победы * 100.0 / SUM(Победы) OVER (PARTITION BY Команда),2) as "Процент побед"
   FROM cte
  ORDER BY 2, 3 DESC;  
  
 -- 13. На какой трассе какая команда чаще всего побеждала. Вывести команды и их победы, процент побед. Ранжировать.
 
WITH cte AS(
SELECT c2.name AS "Трасса",
	   cons.name AS "Команда",	
	   SUM(res.position) AS "Победы"
  FROM results res
	   INNER JOIN drivers d USING (driverId)
		     JOIN constructors cons USING (constructorId)
			 JOIN races USING (raceId)
			 JOIN circuits c2 USING (circuitId)
 WHERE races.YEAR >= 1980 
   AND races.YEAR < 2024
   AND res.position = '1'
 GROUP BY 1,2
)

SELECT DENSE_RANK() OVER (PARTITION BY Трасса ORDER BY Победы DESC) AS "R",
       Трасса,
 	   Команда,
       Победы,
       SUM(Победы) OVER (PARTITION BY Трасса) as "Общее число побед",
       ROUND(Победы * 100.0 / SUM(Победы) OVER (PARTITION BY Трасса),2) as "Процент побед"
  FROM cte
 ORDER BY 2,4 DESC; 

 -- 14. Подсчитать количество очков каждого пилота. Вывести сумму очков команды в КК. Указать долю каждого пилота в сумме. Взять 2022 год.
 
WITH cte AS(   
SELECT CONCAT(d.forename,' ', d.surname) AS "Пилот",
       cons.name AS "Команда",
       SUM(res.points) AS "Очки"
  FROM results res
       INNER JOIN drivers d USING (driverId)
			 JOIN constructors cons USING (constructorId)
			 JOIN races USING (raceId)
 WHERE races.YEAR > 2021 
   AND races.YEAR < 2023
 GROUP BY 1,2
 ORDER BY 2
 )
          
 SELECT Команда,
        Пилот,
        Очки,
        ROUND(Очки * 100.0 / SUM(Очки) OVER (PARTITION BY Команда),2) as "%,очков",
        SUM(Очки) OVER (PARTITION BY Команда) as "Очки КК"
   FROM cte
  ORDER BY 5 DESC;

-- 15. Сколько всего пилотов выступало за команду

WITH cte AS( 
SELECT DISTINCT CONCAT(d.forename,' ', d.surname) AS "Пилоты",
       cons.name AS "Команда"
  FROM results res
       INNER JOIN drivers d USING (driverId)
			 JOIN constructors cons USING (constructorId)
 ORDER BY 2 DESC
 )
 
SELECT Команда,
 	   COUNT(Пилоты)
  FROM cte
 GROUP BY 1
 ORDER BY 2 DESC;

-- 16. Вывести круги М.Шумахера в Спа в 2004. Сравнить каждый круг с предыдущим. Построить скользящее среднее с окном равным 5.

WITH cte AS(
SELECT l2.lap, 
       r2.YEAR,
       c2.name,
       time_format(SEC_TO_TIME(milliseconds/1000), '%i:%s:%f') AS "foll_time",
       LAG(time_format(SEC_TO_TIME(milliseconds/1000), '%i:%s:%f'), 1) OVER (ORDER BY l2.lap) AS "prev_time",
       milliseconds - LAG(milliseconds, 1) OVER (ORDER BY l2.lap) AS "diff",
       ROUND(AVG(milliseconds) OVER (ORDER BY l2.lap
       								  ROWS BETWEEN 1 PRECEDING 
       								  AND 2 FOLLOWING)) AS "roll_avg"
  FROM laptimes l2
       INNER JOIN races r2 USING (raceId)
       		 JOIN circuits c2 USING (circuitId)
             JOIN drivers d USING (driverId)
  WHERE c2.name = 'Albert Park Grand Prix Circuit'
    AND d.surname = 'Schumacher' 
    AND d.forename = 'Michael'
    AND r2.YEAR = 2004
ORDER BY 1 ASC
)

SELECT cte.lap, cte.name, foll_time, prev_time,
	time_format(SEC_TO_TIME(diff/1000), '%i:%s:%f') AS "Разница",
	time_format(SEC_TO_TIME(roll_avg/1000), '%i:%s:%f') AS "Скользящее"
	FROM cte
 
	
-- 17. Вывести страны, на чьих трассах чаще всего проводили гонки.