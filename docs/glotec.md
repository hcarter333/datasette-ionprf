# Use Cases 

## Update glotec database
The glotec database can be updated to specific, new interval of time using 
 
```python glotec.py -nmpatch 2025-08-19T00:00:00Z 2025-08-21T00:00:00Z``` 

```python glotec.py -outing_tw 2025-08-19T00:00:00Z 2025-08-21T00:00:00Z```  
where the first timestamp should be at least one minute before the earliest qso.
