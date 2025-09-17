-- Create table
CREATE TABLE dim_date (
    DateKey INT PRIMARY KEY,                 -- YYYYMMDD
    [Date] DATE NOT NULL,
    Day INT NOT NULL,
    DayOfWeek INT NOT NULL,                   -- 1=Monday, 7=Sunday
    DayName NVARCHAR(20) NOT NULL,
    WeekOfYear INT NOT NULL,
    Month INT NOT NULL,
    MonthName NVARCHAR(20) NOT NULL,
    Quarter INT NOT NULL,
    Year INT NOT NULL,
    FirstOfMonth NVARCHAR(7) NOT NULL,        -- yyyy-01
    MonthYear NVARCHAR(20) NOT NULL,          -- e.g. 'January 2025'
    IsWeekend BIT NOT NULL
);

-- Populate table
DECLARE @StartDate DATE = '2000-01-01';
DECLARE @EndDate   DATE = '2030-12-31';

WITH DateSeries AS (
    SELECT @StartDate AS [Date]
    UNION ALL
    SELECT DATEADD(DAY, 1, [Date])
    FROM DateSeries
    WHERE [Date] < @EndDate
)
INSERT INTO dim_date
SELECT
    CONVERT(INT, FORMAT([Date], 'yyyyMMdd')) AS DateKey,
    [Date],
    DAY([Date]) AS Day,
    DATEPART(WEEKDAY, [Date]) AS DayOfWeek,
    DATENAME(WEEKDAY, [Date]) AS DayName,
    DATEPART(WEEK, [Date]) AS WeekOfYear,
    MONTH([Date]) AS Month,
    DATENAME(MONTH, [Date]) AS MonthName,
    DATEPART(QUARTER, [Date]) AS Quarter,
    YEAR([Date]) AS Year,
    FORMAT(DATEFROMPARTS(YEAR([Date]), 1, 1), 'yyyy-01') AS FirstOfMonth,
    CONCAT(DATENAME(MONTH, [Date]), ' ', YEAR([Date])) AS MonthYear,
    CASE WHEN DATEPART(WEEKDAY, [Date]) IN (1, 7) THEN 1 ELSE 0 END AS IsWeekend
FROM DateSeries
OPTION (MAXRECURSION 0);