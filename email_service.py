#!/usr/bin/env python3
"""
Email notification service for Lost and Found system
"""

from flask_mail import Message
from flask import current_app, url_for
import logging

logger = logging.getLogger(__name__)

class EmailService:
    """Service for sending email notifications"""
    
    def __init__(self, mail):
        self.mail = mail
    
    def send_match_notification(self, recipient_email, recipient_username, user_item_title, matched_item_title, matched_item_owner, matched_item_image_path, match_confidence=1.0):
        """
        Send email notification when a match is found
        
        Args:
            recipient_email: Email address of the recipient
            recipient_username: Username of the recipient
            user_item_title: Title of the user's item
            matched_item_title: Title of the matched item
            matched_item_owner: Username of the matched item owner
            matched_item_image_path: Path to the matched item's image
            match_confidence: Confidence score of the match (0-1)
        """
        try:
            # Check if recipient has email
            if not recipient_email:
                logger.warning(f"Cannot send notification: Recipient {recipient_username} has no email")
                return False
            
            # Prepare email content
            subject = f"🎉 Match Found for Your {user_item_title}!"
            
            # Create HTML email template
            html_body = self._create_match_notification_html(
                recipient_username, user_item_title, matched_item_title, matched_item_owner, matched_item_image_path, match_confidence
            )
            
            # Create text version for email clients that don't support HTML
            text_body = self._create_match_notification_text(
                recipient_username, user_item_title, matched_item_title, matched_item_owner, match_confidence
            )
            
            # Create message
            msg = Message(
                subject=subject,
                recipients=[recipient_email],
                html=html_body,
                body=text_body
            )
            
            # Send email
            self.mail.send(msg)
            logger.info(f"Match notification sent to {lost_owner.email} for item {lost_item.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send match notification: {e}")
            return False
    
    def _create_match_notification_html(self, recipient_username, user_item_title, matched_item_title, matched_item_owner, matched_item_image_path, match_confidence):
        """Create HTML email template for match notification"""
        
        # Format confidence as percentage
        confidence_percent = f"{match_confidence * 100:.1f}%"
        
        # Create image HTML if image path is provided
        image_html = ""
        if matched_item_image_path:
            # Use absolute URL for the image
            image_url = f"http://127.0.0.1:4000{url_for('static', filename=matched_item_image_path)}"
            image_html = f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="{image_url}" alt="{matched_item_title}" style="max-width: 300px; max-height: 300px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Item Match Found</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: #f8f9fa;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                }}
                .match-card {{
                    background: white;
                    border: 2px solid #28a745;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .confidence-badge {{
                    background: #28a745;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-weight: bold;
                    display: inline-block;
                    margin: 10px 0;
                }}
                .contact-info {{
                    background: #e3f2fd;
                    border-left: 4px solid #2196f3;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .button {{
                    background: #007bff;
                    color: white;
                    padding: 12px 30px;
                    text-decoration: none;
                    border-radius: 5px;
                    display: inline-block;
                    margin: 10px 5px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    color: #666;
                    font-size: 14px;
                }}
                .item-details {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎉 Great News!</h1>
                <p>A potential match has been found for your lost item!</p>
            </div>
            
            <div class="content">
                <div class="match-card">
                    <h2>Match Details</h2>
                    <div class="confidence-badge">Match Confidence: {confidence_percent}</div>
                    
                    <h3>Your Item:</h3>
                    <div class="item-details">
                        <strong>Item:</strong> {user_item_title}
                    </div>
                    
                    <h3>Matched Item:</h3>
                    <div class="item-details">
                        <strong>Item:</strong> {matched_item_title}<br>
                        <strong>Owner:</strong> {matched_item_owner}
                    </div>
                    
                    {image_html}
                </div>
                
                <div class="contact-info">
                    <h3>📞 Contact Information</h3>
                    <p><strong>Matched Item Owner:</strong> {matched_item_owner}</p>
                    <p><em>Please contact them to verify the match and arrange for item return.</em></p>
                </div>
                
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h4>⚠️ Important Notes:</h4>
                    <ul>
                        <li>Please verify the item details before meeting with the owner</li>
                        <li>Meet in a safe, public location if arranging a pickup</li>
                        <li>Consider offering a small reward as a token of appreciation</li>
                        <li>If this is not your item, please mark it as "Not a Match" in the system</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>This is an automated message from the Lost and Found system.</p>
                <p>If you believe this is an error, please contact support.</p>
            </div>
        </body>
        </html>
        """
        return html
    
    def _create_match_notification_text(self, recipient_username, user_item_title, matched_item_title, matched_item_owner, match_confidence):
        """Create text version of match notification email"""
        
        confidence_percent = f"{match_confidence * 100:.1f}%"
        
        text = f"""
🎉 GREAT NEWS! A match has been found for your item!

MATCH DETAILS:
Match Confidence: {confidence_percent}

YOUR ITEM:
- Item: {user_item_title}

MATCHED ITEM:
- Item: {matched_item_title}
- Owner: {matched_item_owner}

CONTACT INFORMATION:
- Matched Item Owner: {matched_item_owner}

Please contact them to verify the match and arrange for item return.

IMPORTANT NOTES:
- Please verify the item details before meeting with the owner
- Meet in a safe, public location if arranging a pickup
- Consider offering a small reward as a token of appreciation
- If this is not your item, please mark it as "Not a Match" in the system

This is an automated message from the Lost and Found system.
If you believe this is an error, please contact support.
        """
        return text.strip()

# Create global email service instance
email_service = None

def init_email_service(mail):
    """Initialize the global email service"""
    global email_service
    email_service = EmailService(mail)

def get_email_service():
    """Get the global email service instance"""
    return email_service
