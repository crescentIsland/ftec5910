# Online Auctions data source

Source: https://www.kaggle.com/datasets/onlineauctions/online-auctions-dataset

Provider API: https://www.kaggle.com/api/v1/datasets/view/onlineauctions/online-auctions-dataset

Provider license field: Unknown, checked 2026-10-21. Provider credits the companion website for *Modeling Online Auctions* by Wolfgang Jank and Galit Shmueli (2010): http://www.modelingonlineauctions.com/datasets .

Download locally with `python scripts/download_online_auctions.py`. The raw input is intentionally not tracked.

Expected path: `data/raw/online_auctions/auction.csv`

Analyzed raw SHA-256:
`c05d10bafd4f33aadb2c70ab562a23937bd11c8d32a7c73244c829c06302486e`

Fields: auctionid, bid, bidtime, bidder, bidderrate, openbid, price, item, auction_type.

The provider describes bid as a proxy bid, bidtime as days from the start, bidderrate as bidder feedback, openbid as opening price, and price as closing price. Do not treat a record as an independent auction or bidder feedback as seller reputation. The file includes bidder handles, including email-like values; source identifiers are not needed in published aggregate results.

Raw bidder IDs, per-auction derived records and the duplicate standalone Xbox source remain local. Summary metrics and anonymised figures are tracked.
