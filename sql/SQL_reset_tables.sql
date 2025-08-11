-- RESET THE WHOLE DATABASE

TRUNCATE TABLE dbo.dim_version;

-- Drop all dimension tables
DROP TABLE IF EXISTS dbo.dim_department;
DROP TABLE IF EXISTS dbo.dim_customer;
DROP TABLE IF EXISTS dbo.dim_product;
DROP TABLE IF EXISTS dbo.dim_account;
DROP TABLE IF EXISTS dbo.dim_procurement;
DROP TABLE IF EXISTS dbo.dim_service;
DROP TABLE IF EXISTS dbo.dim_line;
DROP TABLE IF EXISTS dbo.dim_employee;
DROP TABLE IF EXISTS dbo.dim_vendor;

-- Drop all fact tables
DROP TABLE IF EXISTS dbo.fact_general_ledger;
DROP TABLE IF EXISTS dbo.fact_payroll;
DROP TABLE IF EXISTS dbo.fact_employee;
