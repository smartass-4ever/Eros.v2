"""
CNS Stripe Billing Integration
Handles payment processing, subscriptions, and usage-based billing
"""

import stripe
import os
import json
from typing import Dict, Optional
from datetime import datetime
from cns_enterprise_api_keys import APITier


class CNSBillingManager:
    """Manages Stripe billing for CNS Enterprise API"""
    
    PRICING = {
        APITier.FREE: {
            'monthly_fee': 0,
            'price_per_1k_requests': 0,
            'included_requests': 1000
        },
        APITier.PRO: {
            'monthly_fee': 99,
            'price_per_1k_requests': 0.05,
            'included_requests': 10000
        },
        APITier.ENTERPRISE: {
            'monthly_fee': 999,
            'price_per_1k_requests': 0.02,
            'included_requests': 100000
        }
    }
    
    def __init__(self, stripe_secret_key: str = None):
        if stripe_secret_key is None:
            stripe_secret_key = os.environ.get('STRIPE_SECRET_KEY', '')
        
        if not stripe_secret_key:
            print("⚠️  Warning: STRIPE_SECRET_KEY not set - billing disabled")
            self.enabled = False
        else:
            stripe.api_key = stripe_secret_key
            self.enabled = True
            print("✅ Stripe billing enabled")
        
        self.billing_data_file = 'data/billing_data.json'
        self.billing_data = self._load_billing_data()
    
    def _load_billing_data(self) -> Dict:
        """Load billing data from disk"""
        try:
            with open(self.billing_data_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                'customers': {},
                'subscriptions': {},
                'invoices': []
            }
    
    def _save_billing_data(self):
        """Save billing data to disk"""
        os.makedirs(os.path.dirname(self.billing_data_file), exist_ok=True)
        with open(self.billing_data_file, 'w') as f:
            json.dump(self.billing_data, f, indent=2)
    
    def create_customer(self, 
                       key_id: str,
                       email: str,
                       company_name: str,
                       tier: APITier) -> Optional[str]:
        """
        Create a Stripe customer for a new API key
        
        Returns:
            customer_id: Stripe customer ID or None if billing disabled
        """
        if not self.enabled:
            return None
        
        try:
            customer = stripe.Customer.create(
                email=email,
                name=company_name,
                metadata={
                    'cns_key_id': key_id,
                    'tier': tier.value
                }
            )
            
            self.billing_data['customers'][key_id] = {
                'stripe_customer_id': customer.id,
                'email': email,
                'company_name': company_name,
                'tier': tier.value,
                'created_at': datetime.now().isoformat()
            }
            self._save_billing_data()
            
            return customer.id
            
        except stripe.error.StripeError as e:
            print(f"❌ Stripe error creating customer: {e}")
            return None
    
    def create_subscription(self,
                          key_id: str,
                          tier: APITier,
                          payment_method_id: str = None) -> Optional[Dict]:
        """
        Create a subscription for Pro or Enterprise tier
        
        Args:
            key_id: CNS API key ID
            tier: API tier (PRO or ENTERPRISE)
            payment_method_id: Stripe payment method ID
        
        Returns:
            Subscription details or None if failed
        """
        if not self.enabled or tier == APITier.FREE:
            return None
        
        customer_data = self.billing_data['customers'].get(key_id)
        if not customer_data:
            print(f"❌ Customer not found for key {key_id}")
            return None
        
        try:
            pricing = self.PRICING[tier]
            
            price = stripe.Price.create(
                unit_amount=int(pricing['monthly_fee'] * 100),
                currency='usd',
                recurring={'interval': 'month'},
                product_data={
                    'name': f'CNS {tier.value.capitalize()} Plan',
                    'metadata': {
                        'tier': tier.value,
                        'included_requests': pricing['included_requests']
                    }
                }
            )
            
            subscription_params = {
                'customer': customer_data['stripe_customer_id'],
                'items': [{'price': price.id}],
                'metadata': {
                    'cns_key_id': key_id,
                    'tier': tier.value
                }
            }
            
            if payment_method_id:
                subscription_params['default_payment_method'] = payment_method_id
            
            subscription = stripe.Subscription.create(**subscription_params)
            
            self.billing_data['subscriptions'][key_id] = {
                'stripe_subscription_id': subscription.id,
                'tier': tier.value,
                'status': subscription.status,
                'created_at': datetime.now().isoformat(),
                'current_period_start': subscription.current_period_start,
                'current_period_end': subscription.current_period_end
            }
            self._save_billing_data()
            
            return {
                'subscription_id': subscription.id,
                'status': subscription.status,
                'current_period_end': subscription.current_period_end
            }
            
        except stripe.error.StripeError as e:
            print(f"❌ Stripe error creating subscription: {e}")
            return None
    
    def record_usage(self, key_id: str, requests_count: int):
        """
        Record API usage for metered billing
        
        Args:
            key_id: CNS API key ID
            requests_count: Number of requests to bill
        """
        if not self.enabled:
            return
        
        subscription_data = self.billing_data['subscriptions'].get(key_id)
        if not subscription_data:
            return
        
        try:
            subscription_item_id = subscription_data.get('subscription_item_id')
            if subscription_item_id:
                stripe.SubscriptionItem.create_usage_record(
                    subscription_item_id,
                    quantity=requests_count,
                    timestamp=int(datetime.now().timestamp())
                )
        except stripe.error.StripeError as e:
            print(f"❌ Error recording usage: {e}")
    
    def calculate_usage_charges(self, key_id: str, total_requests: int, tier: APITier) -> Dict:
        """
        Calculate usage-based charges for a billing period
        
        Returns:
            {
                'monthly_fee': float,
                'included_requests': int,
                'overage_requests': int,
                'overage_charge': float,
                'total_charge': float
            }
        """
        pricing = self.PRICING[tier]
        
        monthly_fee = pricing['monthly_fee']
        included_requests = pricing['included_requests']
        overage_requests = max(0, total_requests - included_requests)
        
        overage_charge = (overage_requests / 1000) * pricing['price_per_1k_requests']
        total_charge = monthly_fee + overage_charge
        
        return {
            'monthly_fee': monthly_fee,
            'included_requests': included_requests,
            'overage_requests': overage_requests,
            'overage_charge': round(overage_charge, 2),
            'total_charge': round(total_charge, 2),
            'price_per_1k': pricing['price_per_1k_requests']
        }
    
    def create_invoice(self, key_id: str, total_requests: int, tier: APITier) -> Optional[Dict]:
        """
        Create an invoice for usage-based billing
        
        Args:
            key_id: CNS API key ID
            total_requests: Total requests in billing period
            tier: API tier
        
        Returns:
            Invoice details or None if failed
        """
        if not self.enabled or tier == APITier.FREE:
            return None
        
        customer_data = self.billing_data['customers'].get(key_id)
        if not customer_data:
            return None
        
        try:
            charges = self.calculate_usage_charges(key_id, total_requests, tier)
            
            if charges['overage_requests'] > 0:
                stripe.InvoiceItem.create(
                    customer=customer_data['stripe_customer_id'],
                    amount=int(charges['overage_charge'] * 100),
                    currency='usd',
                    description=f"CNS API Overage: {charges['overage_requests']:,} requests @ ${charges['price_per_1k']}/1k"
                )
            
            invoice = stripe.Invoice.create(
                customer=customer_data['stripe_customer_id'],
                auto_advance=True
            )
            
            invoice = stripe.Invoice.finalize_invoice(invoice.id)
            
            invoice_record = {
                'stripe_invoice_id': invoice.id,
                'key_id': key_id,
                'tier': tier.value,
                'total_requests': total_requests,
                'charges': charges,
                'created_at': datetime.now().isoformat(),
                'status': invoice.status
            }
            
            self.billing_data['invoices'].append(invoice_record)
            self._save_billing_data()
            
            return {
                'invoice_id': invoice.id,
                'amount_due': invoice.amount_due / 100,
                'status': invoice.status,
                'invoice_pdf': invoice.invoice_pdf
            }
            
        except stripe.error.StripeError as e:
            print(f"❌ Error creating invoice: {e}")
            return None
    
    def cancel_subscription(self, key_id: str) -> bool:
        """Cancel a subscription when an API key is revoked"""
        if not self.enabled:
            return True
        
        subscription_data = self.billing_data['subscriptions'].get(key_id)
        if not subscription_data:
            return True
        
        try:
            stripe.Subscription.delete(subscription_data['stripe_subscription_id'])
            
            subscription_data['status'] = 'canceled'
            subscription_data['canceled_at'] = datetime.now().isoformat()
            self._save_billing_data()
            
            return True
            
        except stripe.error.StripeError as e:
            print(f"❌ Error canceling subscription: {e}")
            return False
    
    def get_customer_info(self, key_id: str) -> Optional[Dict]:
        """Get billing information for a customer"""
        if not self.enabled:
            return None
        
        customer_data = self.billing_data['customers'].get(key_id)
        subscription_data = self.billing_data['subscriptions'].get(key_id)
        
        if not customer_data:
            return None
        
        return {
            'customer': customer_data,
            'subscription': subscription_data,
            'pricing': self.PRICING.get(APITier[customer_data['tier'].upper()])
        }


billing_manager = CNSBillingManager()
