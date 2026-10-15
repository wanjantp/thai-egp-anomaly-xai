# Data sources

No data are stored in this repository. All inputs are public and can be downloaded again with
`src/egp_download.py`. `DATA_MANIFEST.csv` lists every file used (112 resources): dataset slug, resource ID,
download URL, size, the portal's last-modified date and our download date (5–14 October 2026).

## 1. e-GP contract records — Open Government Data of Thailand (data.go.th)

Source system: e-GP, Comptroller General's Department (กรมบัญชีกลาง), Ministry of Finance.
Portal organization: Digital Government Development Agency (DGA); publisher field "THAILAND GOVERNMENT SPENDING".
Licence: Creative Commons Attribution (as shown on each dataset page).
Cite as: Digital Government Development Agency (2026a) for the contract data, (2026b) for `summary_cgdcontract`, (2026c) for `egpwinner`.
Dataset title (Thai): "ข้อมูลโครงการจัดซื้อจัดจ้างจากระบบการจัดซื้อจัดจ้างภาครัฐ ปีงบประมาณ …"

| Fiscal year (B.E.) | Fiscal year (CE) | Dataset page | Files | Size (GB) |
|---|---|---|---|---|
| 2558 | 2015 | https://data.go.th/dataset/cgd-contract-2558 | 7 | 3.49 |
| 2559 | 2016 | https://data.go.th/dataset/cgd-contract-2559 | 4 | 1.85 |
| 2560 | 2017 | https://data.go.th/dataset/cgd-contract-2560 | 5 | 1.97 |
| 2561 | 2018 | https://data.go.th/dataset/cgd-contract-2561 | 8 | 3.63 |
| 2562 | 2019 | https://data.go.th/dataset/cgd-contract-2562 | 9 | 4.05 |
| 2563 | 2020 | https://data.go.th/dataset/cgd-contract-2563 | 10 | 4.97 |
| 2564 | 2021 | https://data.go.th/dataset/cdg-contract-2564 | 10 | 5.06 |
| 2565 | 2022 | https://data.go.th/dataset/cdg-contract-2565 | 10 | 4.81 |
| 2566 | 2023 | https://data.go.th/dataset/cdg-contract-2566 | 12 | 5.66 |
| 2567 | 2024 | https://data.go.th/dataset/cdg-contract-2567 | 11 | 5.62 |
| 2568 | 2025 | https://data.go.th/dataset/egp-contact-2568 | 10 | 5.00 |

The dataset identifiers are spelled inconsistently on the portal (`cgd-`, `cdg-`, `egp-contact-`);
the downloader also finds them by title through the CKAN API (`/api/3/action/package_search`).

Supporting datasets from the same publisher:

| Content | Dataset page | Files | Use |
|---|---|---|---|
| Project summary by agency, FY2558–2568 | https://data.go.th/dataset/summary_cgdcontract | 11 | Validation of project counts; agency codes |
| Winning-bidder register | https://data.go.th/dataset/egpwinner | 5 | Vendor fields |

Fiscal years run October–September (FY2558 = October 2014 – September 2015).

## 2. Coverage check before FY2558 — Anti-Corruption Organization of Thailand (ACT)

ACT Open Data, winner records (`act_opendata/opendata/master_data/ds_001/001/winner/`), years 2013–2014.
Used only to check whether earlier years could serve as a reference period (they hold about 2% of a normal year).
Not needed to reproduce the results.

## 3. Legal sources (thresholds and procurement methods)

| Document | Date | Where to find |
|---|---|---|
| Prime Minister's Office Regulation on Procurement B.E. 2535 (1992), as amended | 1992 | Royal Gazette, https://ratchakitcha.soc.go.th |
| Procurement Committee circular กค (กวพ) 0421.3/ว 299 — THB 500,000 ceiling for price agreement | 28 Aug 2015 (in force 15 Sep 2015 – 31 Mar 2016) | Comptroller General's Department |
| Procurement Committee circular กค (กวพ) 0421.3/ว 145 — extension to 30 Sep 2016 | 21 Mar 2016 | Comptroller General's Department |
| Ministry of Interior circular มท 0808.2/ว 1856 — local governments | 31 Mar 2016 | Ministry of Interior |
| Public Procurement and Supplies Administration Act B.E. 2560 (2017), s. 56 | in force 23 Aug 2017 | Royal Gazette, https://ratchakitcha.soc.go.th |
| Ministry of Finance Regulation on Public Procurement and Supplies Administration B.E. 2560 (2017), clause 20 | 2017 | Royal Gazette |
| Circular กค (กวจ) 0405.2/ว 123 — selection and specific methods for promoted supplies | 9 Mar 2018 | e-GP website, https://www.gprocurement.go.th |
